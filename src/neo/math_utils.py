"""Utility math functions used across Neo."""
from __future__ import annotations

import math
from decimal import Decimal, InvalidOperation
from typing import Optional, Union

import logging

import numpy as np

_logger = logging.getLogger(__name__)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors.

    Returns 0.0 for zero-norm or non-finite vectors.
    """
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        _logger.debug("cosine_similarity: NaN or Inf values in input vectors")
        return 0.0
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        _logger.debug("cosine_similarity: zero-norm vector")
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


_RECALL_NORMALIZER = 1.0 - math.exp(-1.0)  # ≈ 0.6321, so r=1,t=0 → p=1


def recall_probability(
    cosine: float,
    *,
    days_since_recall: float,
    g_n: float,
) -> float:
    """Ebbinghaus-style recall probability (Hou et al., paper 2404.00573).

        p_n(t) = (1 - exp(-r · exp(-t / g_n))) / (1 - e^-1)

    Where r is cosine similarity, t is elapsed time in days since the last
    recall, and g_n is a per-fact strength that grows with each spaced recall.
    Returned value is clamped to [0, 1].
    """
    if cosine <= 0.0:
        return 0.0
    if g_n <= 0.0:
        _logger.warning("recall_probability: invalid g_n=%r, falling back to 1.0", g_n)
        g_n = 1.0
    inner = cosine * math.exp(-max(0.0, days_since_recall) / g_n)
    p = (1.0 - math.exp(-inner)) / _RECALL_NORMALIZER
    if not math.isfinite(p):
        return 0.0
    return max(0.0, min(1.0, p))


def g_n_update(g_n: float, days_since_last: float) -> float:
    """Spaced-repetition strengthening for g_n.

        g_{n+1} = g_n + (1 - e^-t) / (1 + e^-t)

    Equivalent to ``g_n + tanh(t/2)``. Bounded growth: each recall adds at
    most 1.0 (long gaps) and at least ~0 (immediate re-recall).
    """
    t = max(0.0, days_since_last)
    e = math.exp(-t)
    return g_n + (1.0 - e) / (1.0 + e)


def batched_cosine(
    embeddings: list[Optional[np.ndarray]],
    query: Optional[np.ndarray],
    *,
    default: float = 0.5,
) -> list[float]:
    """Cosine similarity of one query against many embeddings, one numpy pass.

    Rows with None or non-finite embeddings — and the case where ``query`` is
    None or zero-norm — fall back to ``default``. Vectorized: O(n * d) but in
    one matrix-vector product rather than n Python iterations.
    """
    n = len(embeddings)
    if n == 0:
        return []
    if query is None:
        return [default] * n

    q = np.asarray(query, dtype=np.float32)
    q_norm = float(np.linalg.norm(q))
    if q_norm == 0.0 or not np.isfinite(q_norm):
        return [default] * n

    rows: list[np.ndarray] = []
    row_indices: list[int] = []
    for i, e in enumerate(embeddings):
        if e is None:
            continue
        rows.append(e)
        row_indices.append(i)

    sims = [default] * n
    if not rows:
        return sims

    matrix = np.asarray(rows, dtype=np.float32)
    row_norms = np.linalg.norm(matrix, axis=1)
    safe = row_norms.copy()
    safe[safe == 0.0] = 1.0
    dots = matrix @ q
    cos = dots / (safe * q_norm)

    for idx, c, rn in zip(row_indices, cos, row_norms):
        if rn == 0.0 or not np.isfinite(c):
            continue
        sims[idx] = float(c)
    return sims

NumberLike = Union[int, float, Decimal, str]


def add_numbers(a: NumberLike, b: NumberLike, *, bit_limit: Optional[int] = 64):
    """Add two numeric values with validation and overflow protection.

    Parameters
    ----------
    a, b: Union[int, float, Decimal, str]
        Operands to add. Strings must represent finite numeric values.
    bit_limit: Optional[int], optional
        When provided, enforce the operands and result fit within a signed range
        defined by ``bit_limit`` bits. Defaults to 64. Set to ``None`` to disable
        the range check.

    Returns
    -------
    Union[int, Decimal]
        ``int`` when both inputs are integers and the result is integral;
        otherwise a ``Decimal`` preserving precision.

    Raises
    ------
    TypeError
        If either operand is not a supported numeric type or represents a
        non-finite value (NaN/Infinity) or a boolean.
    OverflowError
        If an operand or the result exceeds the permitted bit range.
    ValueError
        If ``bit_limit`` is provided but not a positive integer.
    """

    if bit_limit is not None and bit_limit <= 0:
        raise ValueError("bit_limit must be a positive integer or None")

    operands = (a, b)
    decimals = tuple(_coerce_to_decimal(value) for value in operands)

    if bit_limit is not None:
        bound = Decimal(2) ** (bit_limit - 1)
        min_bound = -bound
        max_bound = bound - 1

        for idx, dec in enumerate(decimals):
            if not min_bound <= dec <= max_bound:
                raise OverflowError(
                    f"Operand {idx}={operands[idx]!r} exceeds +/-{bit_limit}-bit range"
                )
    else:
        min_bound = max_bound = None

    result = decimals[0] + decimals[1]

    if min_bound is not None and not min_bound <= result <= max_bound:
        raise OverflowError(
            f"Result {result} exceeds +/-{bit_limit}-bit range"
        )

    if (
        all(isinstance(value, int) and not isinstance(value, bool) for value in operands)
        and result == result.to_integral_value()
    ):
        return int(result)

    return result.normalize()


def _coerce_to_decimal(value: NumberLike) -> Decimal:
    """Convert supported numeric input to a finite Decimal value."""

    if isinstance(value, bool):
        raise TypeError("Boolean values are not valid numeric operands")

    if isinstance(value, Decimal):
        dec_value = value
    elif isinstance(value, int):
        dec_value = Decimal(value)
    elif isinstance(value, float):
        if not math.isfinite(value):
            raise TypeError("Float operands must be finite")
        dec_value = Decimal(str(value))
    elif isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            raise TypeError("String operands must contain a numeric value")
        try:
            dec_value = Decimal(stripped)
        except InvalidOperation as error:
            raise TypeError(f"Invalid numeric string: {value!r}") from error
    else:
        raise TypeError(f"Unsupported operand type: {type(value).__name__}")

    if not dec_value.is_finite():
        raise TypeError("Operands must represent finite numbers")

    return dec_value
