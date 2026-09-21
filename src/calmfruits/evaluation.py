"""Leakage-safe query split and information-retrieval metrics."""

from __future__ import annotations

from collections.abc import Callable
import math

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from .catalog import clean_query

K_VALUES = (1, 3, 5, 10)
RELEVANCE_THRESHOLD = 2


def split_queries(queries: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    query_table = queries[["query_id", "query_text"]].drop_duplicates("query_id").copy()
    query_table["normalized_query_text"] = query_table.query_text.map(clean_query)
    assert query_table.groupby("query_id").normalized_query_text.nunique().eq(1).all()
    splitter = GroupShuffleSplit(n_splits=1, test_size=.2, random_state=seed)
    dev_index, val_index = next(splitter.split(query_table, groups=query_table.normalized_query_text))
    query_table["split"] = "development"
    query_table.iloc[val_index, query_table.columns.get_loc("split")] = "validation"
    assert set(query_table.iloc[dev_index].normalized_query_text).isdisjoint(set(query_table.iloc[val_index].normalized_query_text))
    return query_table.sort_values("query_id").reset_index(drop=True)


def build_golden_set(queries: pd.DataFrame, split: pd.DataFrame) -> pd.DataFrame:
    validation_ids = split.loc[split["split"].eq("validation"), "query_id"]
    golden = queries.loc[queries.query_id.isin(validation_ids)].copy()
    assert golden.query_id.nunique() == len(validation_ids)
    return golden.sort_values(["query_id", "item_id"]).reset_index(drop=True)


def _dcg(relevances: list[int]) -> float:
    return float(sum((2**rel - 1) / math.log2(position + 2) for position, rel in enumerate(relevances)))


def _query_metrics(ranked_ids: list[int], labels: dict[int, int], catalog_ids: set[int], k_values: tuple[int, ...] = K_VALUES, threshold: int = RELEVANCE_THRESHOLD) -> dict[str, float]:
    relevant_ids = {item_id for item_id, relevance in labels.items() if relevance >= threshold}
    ranked_relevance = [int(labels.get(item_id, 0)) for item_id in ranked_ids]
    result: dict[str, float] = {}
    first = next((rank + 1 for rank, relevance in enumerate(ranked_relevance) if relevance >= threshold), None)
    result["mrr"] = 0.0 if first is None else 1.0 / first
    for k in k_values:
        top = ranked_relevance[:k]
        binary_hits = sum(relevance >= threshold for relevance in top)
        result[f"precision_at_{k}"] = binary_hits / k
        result[f"recall_at_{k}"] = 0.0 if not relevant_ids else binary_hits / len(relevant_ids)
        result[f"hit_rate_at_{k}"] = float(binary_hits > 0)
        ideal = sorted(labels.values(), reverse=True)[:k]
        denominator = _dcg(ideal)
        result[f"ndcg_at_{k}"] = 0.0 if denominator == 0 else _dcg(top) / denominator
    reachable_relevant = relevant_ids & catalog_ids
    result["relevant_items"] = float(len(relevant_ids))
    result["reachable_relevant_items"] = float(len(reachable_relevant))
    result["reachable_relevant_share"] = 0.0 if not relevant_ids else len(reachable_relevant) / len(relevant_ids)
    reachable_ideal = sorted((relevance for item_id, relevance in labels.items() if item_id in catalog_ids), reverse=True)
    full_ideal = sorted(labels.values(), reverse=True)
    result["ndcg_upper_bound_at_10"] = 0.0 if _dcg(full_ideal[:10]) == 0 else _dcg(reachable_ideal[:10]) / _dcg(full_ideal[:10])
    return result


def evaluate_search(search: Callable[[object, int], pd.DataFrame], golden: pd.DataFrame, catalog_ids: pd.Index, method: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    labels = golden.groupby("query_id").apply(lambda part: dict(zip(part.item_id.astype(int), part.relevance.astype(int))), include_groups=False)
    text_by_query = golden.groupby("query_id").query_text.first()
    catalog_id_set = set(catalog_ids.astype(int))
    metric_rows: list[dict[str, object]] = []
    top_rows: list[pd.DataFrame] = []
    for query_id, query_text in text_by_query.items():
        results = search(query_text, len(catalog_ids))
        ranked = results.imt_id.astype(int).tolist()
        metrics = _query_metrics(ranked, labels.loc[query_id], catalog_id_set)
        metrics.update({"method": method, "query_id": query_id, "query_text": query_text})
        metric_rows.append(metrics)
        top = results.head(10).copy()
        top.insert(0, "query_id", query_id)
        top.insert(1, "query_text", query_text)
        top.insert(2, "method", method)
        top["relevance"] = top.imt_id.map(labels.loc[query_id]).fillna(0).astype(int)
        top_rows.append(top)
    return pd.DataFrame(metric_rows), pd.concat(top_rows, ignore_index=True)


def summarize_metrics(per_query: pd.DataFrame) -> pd.DataFrame:
    metric_columns = [column for column in per_query.columns if column in {"mrr"} or column.startswith(("precision_at_", "recall_at_", "hit_rate_at_", "ndcg_at_"))]
    return per_query.groupby("method")[metric_columns].mean().reset_index()
