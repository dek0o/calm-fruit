"""Week-three retrieval variants built on the frozen week-two protocol."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

from .catalog import clean_query
from .search import RESULT_COLUMNS, SemanticSearch, _empty_result, _ranked_result, _validate_top_k


RRF_K = 60
PROJECTION_DIMENSION = 384
PROJECTION_MARGIN = 0.2
PROJECTION_REGULARIZATION = 0.001
PROJECTION_LEARNING_RATE = 0.001
PROJECTION_BATCH_SIZE = 64
PROJECTION_EPOCHS = 20


class HybridSearch:
    """RRF fusion of independently produced lexical and semantic rankings."""

    def __init__(self, lexical, semantic, alpha: float) -> None:
        if not 0 < alpha < 1:
            raise ValueError("alpha must be strictly between 0 and 1")
        if not lexical.catalog.imt_id.equals(semantic.catalog.imt_id):
            raise ValueError("Hybrid components must share the exact catalog order")
        self.lexical = lexical
        self.semantic = semantic
        self.catalog = lexical.catalog
        self.alpha = float(alpha)

    def search_hybrid(self, query: object, top_k: int) -> pd.DataFrame:
        normalized = clean_query(query)
        if not normalized:
            return _empty_result()
        requested = _validate_top_k(top_k, len(self.catalog))
        lexical_ids = self.lexical.search_lexical(normalized, len(self.catalog)).imt_id.to_numpy()
        semantic_ids = self.semantic.search_semantic(normalized, len(self.catalog)).imt_id.to_numpy()
        scores = np.zeros(len(self.catalog), dtype=np.float64)
        index_by_id = pd.Series(np.arange(len(self.catalog)), index=self.catalog.imt_id).to_dict()
        for rank, item_id in enumerate(lexical_ids, start=1):
            scores[index_by_id[int(item_id)]] += (1 - self.alpha) / (RRF_K + rank)
        for rank, item_id in enumerate(semantic_ids, start=1):
            scores[index_by_id[int(item_id)]] += self.alpha / (RRF_K + rank)
        return _ranked_result(self.catalog, scores, requested)


def build_hard_triplets(
    development: pd.DataFrame,
    catalog_ids: pd.Index,
    base_model,
    catalog_embeddings: np.ndarray,
    seed: int,
    max_negatives: int = 5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    """Build only explicit positive/negative labeled triples from development."""
    del seed  # Ranking and all ties are deterministic; no random sampling occurs.
    available = set(catalog_ids.astype(int))
    id_to_position = pd.Series(np.arange(len(catalog_ids)), index=catalog_ids.astype(int)).to_dict()
    queries = development[["query_id", "query_text"]].drop_duplicates("query_id").sort_values("query_id")
    query_vectors = base_model.encode(
        queries.query_text.map(clean_query).tolist(),
        batch_size=32,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    ).astype(np.float32)
    vector_by_query = dict(zip(queries.query_id, query_vectors))
    anchors: list[np.ndarray] = []
    positives: list[np.ndarray] = []
    negatives: list[np.ndarray] = []
    diagnostics: list[dict[str, object]] = []
    for query_id, part in development.groupby("query_id", sort=True):
        positive_ids = sorted(set(part.loc[(part.relevance >= 2) & part.item_id.isin(available), "item_id"].astype(int)))
        negative_ids = sorted(set(part.loc[(part.relevance < 2) & part.item_id.isin(available), "item_id"].astype(int)))
        if not positive_ids or not negative_ids:
            diagnostics.append({"query_id": query_id, "positive_ids": len(positive_ids), "negative_ids": len(negative_ids), "triplets": 0})
            continue
        query_vector = vector_by_query[query_id]
        ordered_negative_ids = sorted(
            negative_ids,
            key=lambda item_id: (-float(query_vector @ catalog_embeddings[id_to_position[item_id]]), item_id),
        )[:max_negatives]
        for positive_id in positive_ids:
            for negative_id in ordered_negative_ids:
                anchors.append(query_vector)
                positives.append(catalog_embeddings[id_to_position[positive_id]])
                negatives.append(catalog_embeddings[id_to_position[negative_id]])
        diagnostics.append({"query_id": query_id, "positive_ids": len(positive_ids), "negative_ids": len(negative_ids), "triplets": len(positive_ids) * len(ordered_negative_ids)})
    if not anchors:
        raise ValueError("No labeled positive/negative development triplets are available")
    return np.stack(anchors), np.stack(positives), np.stack(negatives), pd.DataFrame(diagnostics)


def train_shared_projection(
    anchors: np.ndarray,
    positives: np.ndarray,
    negatives: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Train one identity-initialized projection for queries and products only."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    projection = nn.Linear(PROJECTION_DIMENSION, PROJECTION_DIMENSION, bias=False)
    with torch.no_grad():
        projection.weight.copy_(torch.eye(PROJECTION_DIMENSION))
    optimizer = torch.optim.Adam(projection.parameters(), lr=PROJECTION_LEARNING_RATE)
    identity = torch.eye(PROJECTION_DIMENSION)
    anchor_tensor = torch.from_numpy(anchors)
    positive_tensor = torch.from_numpy(positives)
    negative_tensor = torch.from_numpy(negatives)
    generator = torch.Generator().manual_seed(seed)
    loss_rows: list[dict[str, float]] = []
    for epoch in range(1, PROJECTION_EPOCHS + 1):
        permutation = torch.randperm(len(anchor_tensor), generator=generator)
        losses: list[float] = []
        for start in range(0, len(permutation), PROJECTION_BATCH_SIZE):
            batch = permutation[start : start + PROJECTION_BATCH_SIZE]
            anchor = F.normalize(projection(anchor_tensor[batch]), dim=1)
            positive = F.normalize(projection(positive_tensor[batch]), dim=1)
            negative = F.normalize(projection(negative_tensor[batch]), dim=1)
            positive_distance = 1 - torch.sum(anchor * positive, dim=1)
            negative_distance = 1 - torch.sum(anchor * negative, dim=1)
            triplet = F.relu(positive_distance - negative_distance + PROJECTION_MARGIN).mean()
            regularization = PROJECTION_REGULARIZATION * torch.mean((projection.weight - identity) ** 2)
            loss = triplet + regularization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        loss_rows.append({"epoch": epoch, "loss": float(np.mean(losses))})
    weights = projection.weight.detach().numpy().astype(np.float32)
    if not np.isfinite(weights).all():
        raise ValueError("Projection contains non-finite weights")
    return weights, pd.DataFrame(loss_rows)


def apply_projection(vectors: np.ndarray, weights: np.ndarray) -> np.ndarray:
    projected = np.asarray(vectors, dtype=np.float32) @ np.asarray(weights, dtype=np.float32).T
    norms = np.linalg.norm(projected, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("Projection generated a zero vector")
    return (projected / norms).astype(np.float32)


class ProjectedSemanticSearch:
    """Exact search using a frozen MiniLM and an independently trained projection."""

    def __init__(self, catalog: pd.DataFrame, base_model, projected_embeddings: np.ndarray, weights: np.ndarray):
        self.catalog = catalog.reset_index(drop=True).copy()
        self.base_model = base_model
        self.projected_embeddings = np.asarray(projected_embeddings, dtype=np.float32)
        self.weights = np.asarray(weights, dtype=np.float32)
        assert self.catalog.imt_id.is_unique
        assert self.projected_embeddings.shape == (len(self.catalog), PROJECTION_DIMENSION)
        assert self.weights.shape == (PROJECTION_DIMENSION, PROJECTION_DIMENSION)

    def search_projected(self, query: object, top_k: int) -> pd.DataFrame:
        normalized = clean_query(query)
        if not normalized:
            return _empty_result()
        vector = self.base_model.encode([normalized], batch_size=1, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
        projected_query = apply_projection(vector, self.weights)[0]
        return _ranked_result(self.catalog, self.projected_embeddings @ projected_query, top_k)

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / "projection_weights.npy", self.weights)
        np.save(directory / "projection_embeddings.npy", self.projected_embeddings)
