"""Content fingerprints used to bind indexes, splits, and final results."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


def dataframe_fingerprint(frame: pd.DataFrame, columns: list[str]) -> str:
    payload = frame.loc[:, columns].to_csv(index=False, lineterminator="\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def file_fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_manifest(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
