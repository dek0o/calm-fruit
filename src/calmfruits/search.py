"""Search engines with one stable query-to-item contract."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Final

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer

from .catalog import clean_query, catalog_manifest

RESULT_COLUMNS: Final[list[str]] = ["rank", "imt_id", "score", "imt_name", "subj_name", "description"]
TFIDF_CONFIG: Final[dict[str, object]] = {"ngram_range": (1, 2), "max_features": 50_000, "lowercase": True, "norm": "l2"}
SEMANTIC_MODEL: Final[str] = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
SEMANTIC_REVISION: Final[str] = "e8f8c211226b894fcb81acc59f3b34ba3efd5f42"
SEMANTIC_BATCH_SIZE: Final[int] = 32


def _validate_top_k(top_k: int, catalog_size: int) -> int:
    if isinstance(top_k, bool) or not isinstance(top_k, (int, np.integer)) or top_k <= 0:
        raise ValueError("top_k must be a positive integer")
    return min(int(top_k), catalog_size)


def _ranked_result(catalog: pd.DataFrame, scores: np.ndarray, top_k: int) -> pd.DataFrame:
    requested = _validate_top_k(top_k, len(catalog))
    order = np.lexsort((catalog.imt_id.to_numpy(), -scores))[:requested]
    result = catalog.iloc[order][["imt_id", "imt_name", "subj_name", "description"]].copy()
    result.insert(0, "rank", np.arange(1, len(result) + 1, dtype=np.int64))
    result.insert(2, "score", scores[order].astype(np.float32))
    return result.loc[:, RESULT_COLUMNS].reset_index(drop=True)


def _empty_result() -> pd.DataFrame:
    return pd.DataFrame({"rank": pd.Series(dtype="int64"), "imt_id": pd.Series(dtype="int64"), "score": pd.Series(dtype="float32"), "imt_name": pd.Series(dtype="string"), "subj_name": pd.Series(dtype="string"), "description": pd.Series(dtype="string")})


class LexicalSearch:
    """TF-IDF retrieval over the fixed evaluation catalog."""

    def __init__(self, catalog: pd.DataFrame, vectorizer: TfidfVectorizer, matrix: sparse.csr_matrix):
        self.catalog = catalog.reset_index(drop=True).copy()
        self.vectorizer = vectorizer
        self.matrix = matrix.tocsr()
        assert self.catalog.imt_id.is_unique and self.matrix.shape[0] == len(self.catalog)

    @classmethod
    def fit(cls, catalog: pd.DataFrame) -> "LexicalSearch":
        vectorizer = TfidfVectorizer(**TFIDF_CONFIG)
        matrix = vectorizer.fit_transform(catalog.product_text)
        return cls(catalog, vectorizer, matrix)

    def search_lexical(self, query: object, top_k: int) -> pd.DataFrame:
        normalized = clean_query(query)
        if not normalized:
            return _empty_result()
        vector = self.vectorizer.transform([normalized])
        scores = (self.matrix @ vector.T).toarray().ravel()
        return _ranked_result(self.catalog, scores, top_k)

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.vectorizer, directory / "tfidf_vectorizer.joblib")
        sparse.save_npz(directory / "tfidf_matrix.npz", self.matrix)


class SemanticSearch:
    """Exact normalized embedding retrieval over the fixed evaluation catalog."""

    def __init__(self, catalog: pd.DataFrame, model: SentenceTransformer, embeddings: np.ndarray):
        self.catalog = catalog.reset_index(drop=True).copy()
        self.model = model
        self.embeddings = np.asarray(embeddings, dtype=np.float32)
        assert self.catalog.imt_id.is_unique and self.embeddings.shape[0] == len(self.catalog)
        assert np.allclose(np.linalg.norm(self.embeddings, axis=1), 1, atol=1e-4)

    @classmethod
    def fit(cls, catalog: pd.DataFrame, model_name: str = SEMANTIC_MODEL, batch_size: int = SEMANTIC_BATCH_SIZE) -> "SemanticSearch":
        model = SentenceTransformer(model_name, revision=SEMANTIC_REVISION, device="cpu")
        embeddings = model.encode(catalog.product_text.tolist(), batch_size=batch_size, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False).astype(np.float32)
        return cls(catalog, model, embeddings)

    def search_semantic(self, query: object, top_k: int) -> pd.DataFrame:
        normalized = clean_query(query)
        if not normalized:
            return _empty_result()
        vector = self.model.encode([normalized], batch_size=1, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)[0].astype(np.float32)
        return _ranked_result(self.catalog, self.embeddings @ vector, top_k)

    def truncation_statistics(self, texts: list[str]) -> pd.DataFrame:
        tokenizer = self.model.tokenizer
        lengths = [len(tokenizer(text, add_special_tokens=True, truncation=False)["input_ids"]) for text in texts]
        limit = int(self.model.max_seq_length)
        return pd.DataFrame({"texts": [len(lengths)], "max_seq_length": [limit], "median_tokens": [float(np.median(lengths))], "p95_tokens": [float(np.quantile(lengths, .95))], "max_tokens": [int(max(lengths, default=0))], "truncated_share": [float(np.mean(np.asarray(lengths) > limit))]})

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / "semantic_embeddings.npy", self.embeddings)
        (directory / "semantic_model.json").write_text(json.dumps({"model_name": SEMANTIC_MODEL, "revision": SEMANTIC_REVISION, "batch_size": SEMANTIC_BATCH_SIZE, "embedding_dimension": int(self.embeddings.shape[1])}, indent=2), encoding="utf-8")


def save_index_manifest(catalog: pd.DataFrame, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "index_manifest.json").write_text(json.dumps({"catalog": catalog_manifest(catalog), "tfidf": TFIDF_CONFIG, "semantic_model": SEMANTIC_MODEL, "semantic_revision": SEMANTIC_REVISION, "semantic_batch_size": SEMANTIC_BATCH_SIZE}, ensure_ascii=False, indent=2), encoding="utf-8")
