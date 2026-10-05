"""Snapshot: immutable flat dict of scalar metrics from one simulation run."""

from __future__ import annotations

from collections.abc import Iterator


class Snapshot:
    """Immutable record of scalar metrics from a single replication.

    Keys use dot notation, e.g. ``"wait.mean"``, ``"arrivals.count"``,
    ``"queue_len.time_mean"``.

    Parameters
    ----------
    metrics:
        Flat ``{str: float}`` dict produced by :class:`Stats`.
    meta:
        Optional metadata (replication index, seed, …).
    """

    def __init__(
        self,
        metrics: dict[str, float],
        meta: dict[str, object] | None = None,
    ) -> None:
        self._metrics: dict[str, float] = dict(metrics)
        self.meta: dict[str, object] = dict(meta or {})

    # ------------------------------------------------------------------
    # Mapping-like interface (read-only)
    # ------------------------------------------------------------------

    def __getitem__(self, key: str) -> float:
        return self._metrics[key]

    def __contains__(self, key: object) -> bool:
        return key in self._metrics

    def __iter__(self) -> Iterator[str]:
        return iter(self._metrics)

    def __len__(self) -> int:
        return len(self._metrics)

    def keys(self):
        return self._metrics.keys()

    def values(self):
        return self._metrics.values()

    def items(self):
        return self._metrics.items()

    def get(self, key: str, default: float | None = None) -> float | None:
        return self._metrics.get(key, default)

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, float]:
        """Return a plain copy of the metrics dict."""
        return dict(self._metrics)

    def prefix(self, p: str) -> Snapshot:
        """Return a new Snapshot with all keys prefixed by *p* + '.'."""
        return Snapshot({f"{p}.{k}": v for k, v in self._metrics.items()}, self.meta)

    def __repr__(self) -> str:
        pairs = ", ".join(f"{k}={v:.6g}" for k, v in list(self._metrics.items())[:4])
        ellipsis = ", ..." if len(self._metrics) > 4 else ""
        return f"Snapshot({{{pairs}{ellipsis}}})"
