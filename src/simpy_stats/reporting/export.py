"""Export utilities: CSV and JSON serialization of Snapshots."""

from __future__ import annotations

import csv
import io
import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .snapshot import Snapshot


def to_csv(snapshots: "list[Snapshot] | Snapshot", file=None) -> str | None:
    """Serialize one or more snapshots to CSV.

    Parameters
    ----------
    snapshots:
        A single :class:`Snapshot` or a list of them (one row per snapshot).
    file:
        File-like object to write to.  If ``None``, returns the CSV string.
    """
    if not isinstance(snapshots, list):
        snapshots = [snapshots]

    if not snapshots:
        return "" if file is None else None

    # Collect all keys (ordered by first occurrence)
    all_keys: list[str] = []
    seen: set[str] = set()
    for snap in snapshots:
        for k in snap.keys():
            if k not in seen:
                all_keys.append(k)
                seen.add(k)

    buf = io.StringIO() if file is None else file
    writer = csv.DictWriter(buf, fieldnames=all_keys, extrasaction="ignore")
    writer.writeheader()
    for snap in snapshots:
        writer.writerow({k: snap.get(k, float("nan")) for k in all_keys})

    if file is None:
        return buf.getvalue()
    return None


def to_json(
    snapshots: "list[Snapshot] | Snapshot",
    file=None,
    indent: int = 2,
) -> str | None:
    """Serialize one or more snapshots to JSON.

    Parameters
    ----------
    snapshots:
        A single :class:`Snapshot` or a list of them.
    file:
        File-like object to write to.  If ``None``, returns the JSON string.
    indent:
        JSON indentation level.
    """
    if not isinstance(snapshots, list):
        data = snapshots.to_dict()
    else:
        data = [s.to_dict() for s in snapshots]

    if file is None:
        return json.dumps(data, indent=indent)
    json.dump(data, file, indent=indent)
    return None
