"""Utility math functions used across Neo."""
from __future__ import annotations

import math
from decimal import Decimal, InvalidOperation
from typing import Optional, Union

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
