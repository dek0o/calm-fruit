"""Safe, explicit access to the project sources in Yandex Object Storage."""

from __future__ import annotations

from io import BytesIO
import os
from pathlib import Path
from typing import Final

import boto3
from botocore.config import Config
import pandas as pd

DEFAULT_ENDPOINT: Final[str] = "https://storage.yandexcloud.net"
DEFAULT_BUCKET: Final[str] = "s3-ds-source"
EDA_SOURCE_FILES: Final[frozenset[str]] = frozenset(
    {
        "wb_products_raw_sample.parquet",
        "queries_synthetic_train.parquet",
    }
)
ALL_PROJECT_SOURCE_FILES: Final[frozenset[str]] = frozenset(
    {
        *EDA_SOURCE_FILES,
        "wb_products_dedup.parquet",
        "queries_synthetic_test.parquet",
    }
)


def _required_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(
            f"{name} is not set. Copy .env.example to a local .env or export "
            "the S3 credentials in the shell before running the notebook."
        )
    return value


def load_local_env(path: str | Path = ".env") -> bool:
    """Load a local ``.env`` file without replacing explicitly exported values.

    The tiny parser intentionally supports only ``KEY=value`` records required
    by this project. It never prints values and returns whether a file existed.
    """
    dotenv_path = Path(path)
    if not dotenv_path.exists():
        return False
    for line in dotenv_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", maxsplit=1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value
    return True


def _s3_client():
    """Build a client from environment variables without exposing secrets."""
    session_token = os.getenv("AWS_SESSION_TOKEN")
    client_kwargs: dict[str, str] = {
        "service_name": "s3",
        "endpoint_url": os.getenv("S3_ENDPOINT_URL", DEFAULT_ENDPOINT),
        "aws_access_key_id": _required_env("AWS_ACCESS_KEY_ID"),
        "aws_secret_access_key": _required_env("AWS_SECRET_ACCESS_KEY"),
    }
    if session_token:
        client_kwargs["aws_session_token"] = session_token
    # macOS may expose an invalid loopback proxy through its system settings.
    # The course S3 endpoint must be reached directly; credentials are still
    # supplied only through the process environment or ignored local .env.
    return boto3.client(**client_kwargs, config=Config(proxies={}))


def read_parquet_from_s3(filename: str, *, allowed_files: frozenset[str] = ALL_PROJECT_SOURCE_FILES) -> pd.DataFrame:
    """Read an approved project Parquet object directly from the configured bucket.

    The allow-list prevents a notebook from silently reading the test split during
    a stage where it is forbidden. The function only reads data; it never writes
    source objects back to S3.
    """
    if filename not in allowed_files:
        raise ValueError(f"Unsupported project source: {filename!r}")

    bucket = os.getenv("S3_BUCKET", DEFAULT_BUCKET)
    response = _s3_client().get_object(Bucket=bucket, Key=filename)
    return pd.read_parquet(BytesIO(response["Body"].read()))


def read_eda_sources() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return only sources permitted in the first-week EDA."""
    products = read_parquet_from_s3(
        "wb_products_raw_sample.parquet", allowed_files=EDA_SOURCE_FILES
    )
    queries_train = read_parquet_from_s3(
        "queries_synthetic_train.parquet", allowed_files=EDA_SOURCE_FILES
    )
    return products, queries_train
