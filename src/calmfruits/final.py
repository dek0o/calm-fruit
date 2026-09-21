"""Guarded one-time final evaluation after validation configuration freeze."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .evaluation import _query_metrics, summarize_metrics
from .provenance import dataframe_fingerprint, write_manifest


def assert_final_input_contract(test: pd.DataFrame, train: pd.DataFrame) -> None:
    required = {"query_id", "query_text", "item_id", "relevance"}
    if not required.issubset(test.columns):
        raise ValueError(f"Test split misses columns: {sorted(required - set(test.columns))}")
    if not test.relevance.between(0, 3).all():
        raise ValueError("Test relevance must use the 0–3 scale")
    if set(test.query_id) & set(train.query_id):
        raise ValueError("Test query_id intersects train")
    normalized_train = set(train.query_text.map(lambda text: str(text).strip().lower()))
    normalized_test = set(test.query_text.map(lambda text: str(text).strip().lower()))
    if normalized_train & normalized_test:
        raise ValueError("Test query text intersects train after normalization")


def run_final_once(search, test: pd.DataFrame, catalog_ids: pd.Index, selected_manifest: dict[str, object], output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Evaluate a frozen configuration once, checkpointing each completed query."""
    output_dir.mkdir(parents=True, exist_ok=True)
    state_path = output_dir / "final_test_state.json"
    metrics_path = output_dir / "per_query_metrics.csv"
    top10_path = output_dir / "top10_results.csv"
    summary_path = output_dir / "summary.csv"
    test_fingerprint = dataframe_fingerprint(test, ["query_id", "query_text", "item_id", "relevance"])
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state["selection"] != selected_manifest or state["test_fingerprint"] != test_fingerprint:
            raise RuntimeError("Existing final-test state belongs to a different configuration or test split")
        if state["status"] == "completed":
            return pd.read_csv(metrics_path), pd.read_csv(top10_path), pd.read_csv(summary_path)
    else:
        write_manifest(state_path, {"status": "running", "selection": selected_manifest, "test_fingerprint": test_fingerprint})
    completed = set(pd.read_csv(metrics_path).query_id) if metrics_path.exists() else set()
    labels_by_query = test.groupby("query_id").apply(lambda part: dict(zip(part.item_id.astype(int), part.relevance.astype(int))), include_groups=False)
    text_by_query = test.groupby("query_id").query_text.first()
    catalog_id_set = set(catalog_ids.astype(int))
    for query_id, query_text in text_by_query.items():
        if query_id in completed:
            continue
        result = search(query_text, len(catalog_ids))
        metrics = _query_metrics(result.imt_id.astype(int).tolist(), labels_by_query.loc[query_id], catalog_id_set)
        metrics.update({"method": selected_manifest["method"], "query_id": query_id, "query_text": query_text})
        pd.DataFrame([metrics]).to_csv(metrics_path, mode="a", index=False, header=not metrics_path.exists())
        top = result.head(10).copy()
        top.insert(0, "query_id", query_id)
        top.insert(1, "query_text", query_text)
        top.insert(2, "method", selected_manifest["method"])
        top["relevance"] = top.imt_id.map(labels_by_query.loc[query_id]).fillna(0).astype(int)
        top.to_csv(top10_path, mode="a", index=False, header=not top10_path.exists())
    per_query = pd.read_csv(metrics_path).sort_values("query_id").reset_index(drop=True)
    top10 = pd.read_csv(top10_path).sort_values(["query_id", "rank"]).reset_index(drop=True)
    if per_query.query_id.nunique() != test.query_id.nunique():
        raise RuntimeError("Final test checkpoint is incomplete")
    summary = summarize_metrics(per_query)
    summary.to_csv(summary_path, index=False)
    write_manifest(state_path, {"status": "completed", "selection": selected_manifest, "test_fingerprint": test_fingerprint, "query_count": int(test.query_id.nunique())})
    return per_query, top10, summary
