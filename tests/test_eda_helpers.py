"""Regression tests for deterministic first-week EDA helpers."""

from __future__ import annotations

import sys
from pathlib import Path
import unittest

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


if __name__ == "__main__":
    unittest.main()
