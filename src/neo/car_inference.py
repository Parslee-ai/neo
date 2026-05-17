"""Process-wide singleton for `car_runtime.CarRuntime`.

Both Neo's outbound inference path (``CarAdapter``) and its inbound A2A
host (``car_host.run_server``) should share a single ``CarRuntime``
instance per process so state, policies, the eventlog, and the tool
registry stay consistent across surfaces.

This module is the only place that imports ``car_runtime``. Everywhere
else goes through :func:`get_runtime` (returns the singleton, importing
``car_runtime`` lazily) or :func:`is_available` (cheap check that does
not import the binding).

Tests mock by calling :func:`set_runtime` with a fake.
"""
from __future__ import annotations

import importlib.util
import logging
import threading
from typing import Any, Optional

logger = logging.getLogger(__name__)

_runtime: Optional[Any] = None
_runtime_lock = threading.Lock()


def is_available() -> bool:
    """Return True if ``car_runtime`` can be imported. Does not import it."""
    return importlib.util.find_spec("car_runtime") is not None


def get_runtime() -> Any:
    """Return the process-wide ``CarRuntime`` singleton.

    Instantiates on first call. Raises ``RuntimeError`` with an actionable
    message if the ``[car]`` extra is not installed — the caller is
    responsible for surfacing that to the user.
    """
    global _runtime
    if _runtime is not None:
        return _runtime
    with _runtime_lock:
        if _runtime is not None:
            return _runtime
        try:
            import car_runtime as cr  # type: ignore[import-not-found]
        except ImportError as e:
            raise RuntimeError(
                "car_runtime is not installed. Install Neo's CAR extras:\n"
                "    pip install 'neo-reasoner[car]'\n"
                "and start the CAR daemon (`python -m car_runtime.server` or "
                "`car-server`) before retrying."
            ) from e
        _runtime = cr.CarRuntime()
        logger.info("CarRuntime singleton initialized")
    return _runtime


def set_runtime(runtime: Optional[Any]) -> None:
    """Replace the singleton (test hook). Pass None to reset."""
    global _runtime
    with _runtime_lock:
        _runtime = runtime


def reset_for_testing() -> None:
    """Clear the singleton so the next ``get_runtime()`` call rebuilds it."""
    set_runtime(None)
