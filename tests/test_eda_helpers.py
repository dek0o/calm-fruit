"""Regression tests for deterministic first-week EDA helpers."""

from __future__ import annotations

import sys
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from calmfruits.eda import (  # noqa: E402
    color_mentioned_in_name,
    contains_article_like_token,
    join_text_fields,
    missing_profile,
    select_mvp_root_categories,
)
from calmfruits.evaluation import _query_metrics, build_golden_set, split_queries  # noqa: E402
from calmfruits.catalog import clean_query  # noqa: E402
from calmfruits.experiments import HybridSearch, apply_projection, train_shared_projection  # noqa: E402
from calmfruits.final import run_final_once  # noqa: E402


class EDAHelperTests(unittest.TestCase):
    def test_missing_profile_counts_blank_and_whitespace(self) -> None:
        frame = pd.DataFrame({"text": [None, "", "   ", "valid"]})

        result = missing_profile(frame, ["text"])

        self.assertEqual(result.loc["text", "null_count"], 1)
        self.assertEqual(result.loc["text", "blank_count"], 1)
        self.assertEqual(result.loc["text", "whitespace_count"], 1)
        self.assertEqual(result.loc["text", "missing_total"], 3)

    def test_join_text_fields_omits_missing_values(self) -> None:
        frame = pd.DataFrame({"name": ["Товар", None], "category": ["Обувь", "Одежда"]})

        self.assertEqual(join_text_fields(frame, ["name", "category"]).tolist(), ["Товар . Обувь", "Одежда"])

    def test_article_token_requires_letters_and_digits_in_the_same_token(self) -> None:
        values = pd.Series(["Красивые туфли 42", "Модель ABC123", "123456", "Код_12A"])

        self.assertEqual(contains_article_like_token(values).tolist(), [False, True, False, True])

    def test_color_match_handles_common_inflection(self) -> None:
        frame = pd.DataFrame({"imt_name": ["Кроссовки красные", "Футболка"], "nm_colors_names": ["красный", "белый"]})

        self.assertEqual(color_mentioned_in_name(frame).tolist(), [True, False])

    def test_category_selection_excludes_uncovered_categories(self) -> None:
        products = pd.DataFrame({
            "imt_id": [1, 2, 3],
            "subj_root_name": ["Обувь", "Одежда", "Дом"],
            "imt_name": ["Кроссовки", "Платье", "Лампа"],
            "description": ["Для бега", "Летнее", "Настольная"],
        })
        queries = pd.DataFrame({"query_id": ["q1", "q2"], "item_id": [1, 2], "relevance": [3, 2]})

        result = select_mvp_root_categories(products, queries, top_n=3)

        self.assertEqual(result["subj_root_name"].tolist(), ["Обувь", "Одежда"])

    def test_metrics_preserve_unreachable_relevance_in_recall_and_ndcg(self) -> None:
        metrics = _query_metrics([2, 3], {1: 3, 2: 2}, {2, 3})

        self.assertEqual(metrics["reachable_relevant_share"], 0.5)
        self.assertEqual(metrics["recall_at_1"], 0.5)
        self.assertLess(metrics["ndcg_upper_bound_at_10"], 1)

    def test_group_split_keeps_duplicate_normalized_text_together(self) -> None:
        queries = pd.DataFrame({
            "query_id": ["q1", "q1", "q2", "q2", "q3", "q3", "q4", "q4", "q5", "q5"],
            "query_text": ["Кеды", "Кеды", " кеды ", " кеды ", "Брюки", "Брюки", "Платье", "Платье", "Шуба", "Шуба"],
            "item_id": [1, 2] * 5,
            "relevance": [3, 1] * 5,
        })

        split = split_queries(queries)
        golden = build_golden_set(queries, split)
        development_texts = set(split.loc[split["split"].eq("development"), "normalized_query_text"])
        validation_texts = set(split.loc[split["split"].eq("validation"), "normalized_query_text"])

        self.assertTrue(development_texts.isdisjoint(validation_texts))
        self.assertEqual(golden.query_id.nunique(), split["split"].eq("validation").sum())

    def test_query_normalization_is_case_invariant_for_semantic_inputs(self) -> None:
        self.assertEqual(clean_query("Кроссовки NIKE"), clean_query("кроссовки nike"))

    def test_final_checkpoint_does_not_rerun_completed_test(self) -> None:
        test = pd.DataFrame({"query_id": ["q1", "q1"], "query_text": ["ботинки", "ботинки"], "item_id": [1, 2], "relevance": [3, 1]})
        catalog_ids = pd.Index([1, 2])
        calls = []

        def search(query, top_k):
            calls.append((query, top_k))
            return pd.DataFrame({"rank": [1, 2], "imt_id": [1, 2], "score": [1.0, 0.5], "imt_name": ["Ботинки", "Туфли"], "subj_name": ["Ботинки", "Туфли"], "description": ["", ""]})

        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            first = run_final_once(search, test, catalog_ids, {"method": "lexical"}, directory)
            second = run_final_once(search, test, catalog_ids, {"method": "lexical"}, directory)

        self.assertEqual(len(calls), 1)
        self.assertTrue(first[2].equals(second[2]))

    def test_projection_preserves_unit_norm_and_trains_shared_matrix(self) -> None:
        anchors = np.array([[1.0] + [0.0] * 383, [0.0, 1.0] + [0.0] * 382], dtype="float32")
        positives = anchors.copy()
        negatives = np.array([[0.0, 1.0] + [0.0] * 382, [1.0] + [0.0] * 383], dtype="float32")

        weights, loss_history = train_shared_projection(anchors, positives, negatives, seed=42)
        projected = apply_projection(anchors, weights)

        self.assertEqual(weights.shape, (384, 384))
        self.assertEqual(len(loss_history), 20)
        self.assertTrue(np.isfinite(weights).all())
        self.assertTrue(np.allclose(np.linalg.norm(projected, axis=1), 1))


if __name__ == "__main__":
    unittest.main()
