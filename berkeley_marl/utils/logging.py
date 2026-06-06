from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, Mapping


def append_csv(path: str | Path, row: Mapping[str, object], fieldnames: Iterable[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        if not exists:
            writer.writeheader()
        writer.writerow(row)
