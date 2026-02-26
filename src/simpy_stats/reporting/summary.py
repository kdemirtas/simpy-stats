"""summary_table: human-readable tabular summary of a Snapshot or ReplicationReport."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .snapshot import Snapshot


def summary_table(obj: "Snapshot | dict", title: str = "") -> str:
    """Return a formatted text table of metrics.

    Works with a :class:`Snapshot`, a plain ``{str: float}`` dict, or a
    :class:`~simpy_stats.experiments.replication.ReplicationReport`.

    Parameters
    ----------
    obj:
        Metrics source.
    title:
        Optional heading printed above the table.
    """
    # Support ReplicationReport by extracting its metric summary
    if hasattr(obj, "metric_summary"):
        return _report_table(obj, title)

    if hasattr(obj, "to_dict"):
        data = obj.to_dict()
    else:
        data = dict(obj)

    if not data:
        return "(no metrics)"

    key_w = max(len(k) for k in data) + 2
    lines: list[str] = []
    if title:
        lines.append(title)
        lines.append("=" * max(key_w + 14, len(title)))
    else:
        lines.append("-" * (key_w + 14))

    header = f"{'Metric':<{key_w}}{'Value':>12}"
    lines.append(header)
    lines.append("-" * (key_w + 14))

    for k, v in sorted(data.items()):
        lines.append(f"{k:<{key_w}}{v:>12.6g}")

    lines.append("-" * (key_w + 14))
    return "\n".join(lines)


def _report_table(report, title: str) -> str:  # type: ignore[no-untyped-def]
    """Formatted table for a ReplicationReport."""
    ms = report.metric_summary
    if not ms:
        return "(no metrics in report)"

    cols = ["n", "mean", "stdev", "half_width", "lower", "upper"]
    key_w = max(len(k) for k in ms) + 2
    col_w = 12

    lines: list[str] = []
    if title:
        lines.append(title)
        lines.append("=" * (key_w + col_w * len(cols)))
    else:
        lines.append("-" * (key_w + col_w * len(cols)))

    header = f"{'Metric':<{key_w}}" + "".join(f"{c:>{col_w}}" for c in cols)
    lines.append(header)
    lines.append("-" * (key_w + col_w * len(cols)))

    for metric, stats in sorted(ms.items()):
        row = f"{metric:<{key_w}}"
        for c in cols:
            val = stats.get(c, float("nan"))
            if c == "n":
                row += f"{int(val):>{col_w}}"
            else:
                row += f"{val:>{col_w}.6g}"
        lines.append(row)

    lines.append("-" * (key_w + col_w * len(cols)))
    stop = getattr(report, "stop_reason", None)
    if stop:
        lines.append(f"Stop reason: {stop}")
    return "\n".join(lines)
