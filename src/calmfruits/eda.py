"""Pure analysis helpers used by the first-week EDA notebook."""

from __future__ import annotations

import re
from collections.abc import Iterable

import pandas as pd


def as_clean_string(series: pd.Series) -> pd.Series:
    """Return stripped string values while preserving missing values as ``pd.NA``."""
    return series.astype("string").str.strip().mask(lambda values: values.eq(""))


def missing_profile(frame: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    """Count null, blank and whitespace-only values separately for text fields."""
    rows: list[dict[str, object]] = []
    for column in columns:
        raw = frame[column]
        string_values = raw.astype("string")
        null_mask = raw.isna()
        blank_mask = string_values.eq("").fillna(False)
        whitespace_mask = string_values.str.fullmatch(r"\s+").fillna(False)
        missing_mask = null_mask | blank_mask | whitespace_mask
        rows.append(
            {
                "column": column,
                "null_count": int(null_mask.sum()),
                "blank_count": int(blank_mask.sum()),
                "whitespace_count": int(whitespace_mask.sum()),
                "missing_total": int(missing_mask.sum()),
                "missing_share": float(missing_mask.mean()),
            }
        )
    return pd.DataFrame(rows).set_index("column")


def text_length_profile(series: pd.Series) -> pd.DataFrame:
    """Return word and character length percentiles for nonempty texts."""
    clean = as_clean_string(series).dropna()
    lengths = pd.DataFrame(
        {
            "characters": clean.str.len(),
            "words": clean.str.findall(r"\S+").str.len(),
        }
    )
    return lengths.describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]).T


def contains_html(series: pd.Series) -> pd.Series:
    return as_clean_string(series).str.contains(r"<[^>]+>", regex=True, na=False)


def contains_special_symbols(series: pd.Series) -> pd.Series:
    return as_clean_string(series).str.contains(r"[^\w\s.,;:!?()'\"«»—-]", regex=True, na=False)


def contains_article_like_token(series: pd.Series) -> pd.Series:
    """Flag long alphanumeric tokens as candidates for manual inspection only."""
    pattern = (
        r"(?i)\b"
        r"(?=[a-zа-яё0-9_-]{6,}\b)"
        r"(?=[a-zа-яё0-9_-]*\d)"
        r"(?=[a-zа-яё0-9_-]*[a-zа-яё])"
        r"[a-zа-яё0-9_-]+\b"
    )
    return as_clean_string(series).str.contains(pattern, regex=True, na=False)


def join_text_fields(frame: pd.DataFrame, fields: Iterable[str]) -> pd.Series:
    """Join selected nonempty text fields using an explicit safe separator."""
    candidate_columns = list(fields)
    missing_columns = set(candidate_columns) - set(frame.columns)
    if missing_columns:
        raise ValueError(f"Missing text columns: {sorted(missing_columns)}")
    cleaned = {column: as_clean_string(frame[column]) for column in candidate_columns}
    return pd.concat(cleaned, axis=1).apply(
        lambda row: " . ".join(value for value in row if pd.notna(value)), axis=1
    )


def diagnostic_product_text(frame: pd.DataFrame) -> pd.Series:
    """Build an uncleaned preview of the baseline text without missing-value artifacts."""
    return join_text_fields(frame, ["imt_name", "subj_name", "description"])


def color_mentioned_in_name(frame: pd.DataFrame) -> pd.Series:
    """Identify exact or inflectional color-token overlap for descriptive EDA.

    This is intentionally a diagnostic signal, not a rule for filtering the
    catalog: compound colours and unusual morphology still require the manual
    review included in the notebook.
    """
    names = as_clean_string(frame["imt_name"]).str.lower()
    colors = as_clean_string(frame["nm_colors_names"]).str.lower()

    def stem(token: str) -> str:
        token = token.replace("ё", "е")
        for ending in ("ыми", "ими", "ого", "ему", "ому", "ыми", "ого", "ая", "яя", "ое", "ее", "ой", "ый", "ий", "ые", "ие", "ых", "их", "ым", "им", "ую", "юю", "ом", "ем", "ам", "ям", "ах", "ях", "а", "я", "ы", "и", "у", "ю", "е", "о"):
            if token.endswith(ending) and len(token) - len(ending) >= 4:
                return token[: -len(ending)]
        return token

    def overlaps(row: pd.Series) -> bool | pd.NA:
        name, color_value = row
        if pd.isna(name) or pd.isna(color_value):
            return pd.NA
        normalized_name = name.replace("ё", "е")
        color_tokens = [token.strip() for token in re.split(r"[,/;]", color_value) if token.strip()]
        return any(
            re.search(rf"(?<!\w){re.escape(stem(token))}\w*(?!\w)", normalized_name)
            for token in color_tokens
            if len(stem(token)) >= 4
        )

    return pd.concat([names, colors], axis=1).apply(overlaps, axis=1).astype("boolean")


def select_mvp_root_categories(
    products: pd.DataFrame, queries_train: pd.DataFrame, *, relevance_threshold: int = 2, top_n: int = 3
) -> pd.DataFrame:
    """Rank root categories using relevant train coverage, then indexable catalog size.

    This is a transparent selection rule for the MVP. It does not create an
    evaluation set and it never reads the test split.
    """
    required_products = {"imt_id", "subj_root_name", "imt_name", "description"}
    required_queries = {"query_id", "item_id", "relevance"}
    if not required_products.issubset(products.columns) or not required_queries.issubset(queries_train.columns):
        raise ValueError("Missing columns required for category selection")

    product_lookup = products.loc[:, ["imt_id", "subj_root_name", "imt_name", "description"]].copy()
    product_lookup["subj_root_name"] = as_clean_string(product_lookup["subj_root_name"])
    indexable = as_clean_string(product_lookup["imt_name"]).notna() & as_clean_string(product_lookup["description"]).notna()
    catalog_size = (
        product_lookup.loc[indexable & product_lookup["subj_root_name"].notna()]
        .groupby("subj_root_name")["imt_id"]
        .nunique()
        .rename("indexable_imt_count")
    )
    relevant_queries = queries_train.loc[queries_train["relevance"] >= relevance_threshold, ["query_id", "item_id"]]
    joined = relevant_queries.merge(product_lookup[["imt_id", "subj_root_name"]], left_on="item_id", right_on="imt_id", how="left")
    query_coverage = (
        joined.dropna(subset=["subj_root_name"])
        .groupby("subj_root_name")["query_id"]
        .nunique()
        .rename("relevant_train_query_count")
    )
    result = pd.concat([query_coverage, catalog_size], axis=1).fillna(0).reset_index()
    result["indexable_imt_count"] = result["indexable_imt_count"].astype(int)
    # A category without a relevant training query cannot be evaluated in the
    # MVP and is therefore not an evidence-based candidate for the domain.
    result = result.loc[result["relevant_train_query_count"] > 0]
    return (
        result.sort_values(
            ["relevant_train_query_count", "indexable_imt_count", "subj_root_name"],
            ascending=[False, False, True],
            kind="stable",
        )
        .head(top_n)
        .reset_index(drop=True)
    )
