"""Arm backends: the hardware adapter and a simulator with the same interface."""

from .base import ArmBackend, BackendError

__all__ = ["ArmBackend", "BackendError", "make_backend"]


def make_backend(name: str, **kwargs) -> ArmBackend:
    """Factory: ``"sim"`` or ``"rebotarm"``."""
    if name == "sim":
        from .sim import SimBackend
        return SimBackend(**kwargs)
    if name == "rebotarm":
        from .rebotarm import RebotArmBackend
        return RebotArmBackend(**kwargs)
    raise ValueError(f"unknown backend {name!r} (expected 'sim' or 'rebotarm')")
