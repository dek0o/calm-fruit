"""Deterministic catalog preparation for the CalmFruits MVP."""

from __future__ import annotations

from html import unescape
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

from .data import read_parquet_from_s3
from .eda import join_text_fields

MVP_ROOT_CATEGORIES = ("Одежда", "Обувь")
CATALOG_COLUMNS = (
    "imt_id", "nm_id", "imt_name", "subj_name", "subj_root_name",
    "nm_colors_names", "vendor_code", "description", "brand_name",
)


def clean_text(value: object) -> str:
    """Remove markup/control noise while retaining meaningful technical text."""
    if value is None or pd.isna(value):
        return ""
    text = unescape(str(value))
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[\x00-\x1f\x7f-\x9f]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def clean_query(query: object) -> str:
    """Apply the same non-destructive normalization to products and queries."""
    return clean_text(query).lower()


def _clean_catalog_fields(products: pd.DataFrame) -> pd.DataFrame:
    missing = set(CATALOG_COLUMNS) - set(products.columns)
    if missing:
        raise ValueError(f"Raw catalog misses columns: {sorted(missing)}")
    frame = products.loc[:, CATALOG_COLUMNS].copy()
    frame["source_row"] = np.arange(len(frame), dtype=np.int64)
    for column in ("imt_name", "subj_name", "subj_root_name", "nm_colors_names", "vendor_code", "description", "brand_name"):
        frame[column] = frame[column].map(clean_text)
    return frame


def prepare_catalog(products: pd.DataFrame, evaluation_ids: pd.Series | pd.Index) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build full MVP and evaluation catalogs, reporting every filter stage."""
    frame = _clean_catalog_fields(products)
    stages: list[dict[str, int | str]] = [{"stage": "raw_rows", "rows": len(frame), "unique_imt_id": frame.imt_id.nunique()}]
    frame = frame.loc[frame.subj_root_name.isin(MVP_ROOT_CATEGORIES)].copy()
    stages.append({"stage": "selected_root_categories", "rows": len(frame), "unique_imt_id": frame.imt_id.nunique()})
    frame = frame.loc[frame.imt_name.ne("") & frame.description.ne("")].copy()
    stages.append({"stage": "required_clean_text", "rows": len(frame), "unique_imt_id": frame.imt_id.nunique()})
    frame["description_length"] = frame.description.str.len()
    frame = frame.sort_values(["imt_id", "description_length", "nm_id", "source_row"], ascending=[True, False, True, True], kind="stable")
    full_catalog = frame.drop_duplicates("imt_id", keep="first").drop(columns="description_length").reset_index(drop=True)
    full_catalog["product_text"] = join_text_fields(full_catalog, ["imt_name", "subj_name", "description"])
    assert full_catalog.imt_id.is_unique
    assert full_catalog.product_text.ne("").all()
    assert not full_catalog.product_text.str.contains(r"(?:^| \. )(?:None|nan)(?: \. |$)", case=False, regex=True).any()
    stages.append({"stage": "deduplicated_imt_id", "rows": len(full_catalog), "unique_imt_id": full_catalog.imt_id.nunique()})

    allowed = pd.Index(evaluation_ids).dropna().astype("int64")
    evaluation_catalog = full_catalog.loc[full_catalog.imt_id.isin(allowed)].copy().reset_index(drop=True)
    assert evaluation_catalog.imt_id.is_unique
    stages.append({"stage": "evaluation_catalog_intersection", "rows": len(evaluation_catalog), "unique_imt_id": evaluation_catalog.imt_id.nunique()})
    return full_catalog, evaluation_catalog, pd.DataFrame(stages)


def load_week2_sources() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Read only the raw catalog, train relevance and evaluation-id source."""
    products = read_parquet_from_s3("wb_products_raw_sample.parquet")
    queries = read_parquet_from_s3("queries_synthetic_train.parquet")
    eval_ids = read_parquet_from_s3("wb_products_dedup.parquet")["imt_id"]
    return products, queries, eval_ids


def catalog_manifest(catalog: pd.DataFrame) -> dict[str, object]:
    """Small compatibility record for persisted indexes."""
    return {
        "catalog_size": int(len(catalog)),
        "imt_id_order": catalog.imt_id.astype(int).tolist(),
        "product_text_columns": ["imt_name", "subj_name", "description"],
        "root_categories": list(MVP_ROOT_CATEGORIES),
    }


def save_catalogs(full_catalog: pd.DataFrame, evaluation_catalog: pd.DataFrame, stages: pd.DataFrame, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    full_catalog.to_parquet(directory / "prepared_mvp_catalog.parquet", index=False)
    evaluation_catalog.to_parquet(directory / "evaluation_catalog.parquet", index=False)
    stages.to_csv(directory / "catalog_filter_stages.csv", index=False)
    (directory / "evaluation_catalog_manifest.json").write_text(
        json.dumps(catalog_manifest(evaluation_catalog), ensure_ascii=False, indent=2), encoding="utf-8"
    )
