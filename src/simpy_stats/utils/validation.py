"""Simple validation helpers."""

from __future__ import annotations


def require_positive(value: float, name: str = "value") -> None:
    if value <= 0:
        raise ValueError(f"{name} must be > 0, got {value}")


def require_non_negative(value: float, name: str = "value") -> None:
    if value < 0:
        raise ValueError(f"{name} must be >= 0, got {value}")
