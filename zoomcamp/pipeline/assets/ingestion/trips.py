"""@bruin

name: ingestion.trips
type: python
image: python:3.11
connection: duckdb-default

materialization:
  type: table
  strategy: append

columns:
  - name: taxi_type
    type: string
    description: Taxi type for the source file.
  - name: vendor_id
    type: integer
    description: Taxi vendor identifier.
  - name: pickup_datetime
    type: timestamp
    description: Trip pickup timestamp.
  - name: dropoff_datetime
    type: timestamp
    description: Trip dropoff timestamp.
  - name: passenger_count
    type: integer
    description: Number of passengers.
  - name: trip_distance
    type: float
    description: Trip distance in miles.
  - name: ratecode_id
    type: integer
    description: Rate code identifier.
  - name: store_and_fwd_flag
    type: string
    description: Store and forward flag.
  - name: pickup_location_id
    type: integer
    description: Pickup location identifier.
  - name: dropoff_location_id
    type: integer
    description: Dropoff location identifier.
  - name: payment_type
    type: integer
    description: Payment type identifier.
  - name: fare_amount
    type: float
    description: Base fare amount.
  - name: extra
    type: float
    description: Extra charges.
  - name: mta_tax
    type: float
    description: MTA tax amount.
  - name: tip_amount
    type: float
    description: Tip amount.
  - name: tolls_amount
    type: float
    description: Toll charges.
  - name: improvement_surcharge
    type: float
    description: Improvement surcharge amount.
  - name: total_amount
    type: float
    description: Total charged amount.
  - name: congestion_surcharge
    type: float
    description: Congestion surcharge amount.
  - name: airport_fee
    type: float
    description: Airport fee for eligible trips.
  - name: ehail_fee
    type: float
    description: E-hail fee for green taxis.
  - name: trip_type
    type: integer
    description: Trip type code for green taxis.
  - name: source_url
    type: string
    description: Source parquet file URL.
  - name: extracted_at
    type: timestamp
    description: UTC extraction timestamp.

@bruin"""

from __future__ import annotations

import json
import os
from datetime import date, datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd
import requests
from dateutil.relativedelta import relativedelta

BASE_URL = "https://d37ci6vzurychx.cloudfront.net/trip-data/"
SUPPORTED_TAXI_TYPES = {"yellow", "green"}
LAST_AVAILABLE_MONTH = date(2025, 11, 1)

COMMON_RENAMES = {
    "VendorID": "vendor_id",
    "RatecodeID": "ratecode_id",
    "PULocationID": "pickup_location_id",
    "DOLocationID": "dropoff_location_id",
}

PICKUP_DROPOFF_RENAMES = {
    "yellow": {
        "tpep_pickup_datetime": "pickup_datetime",
        "tpep_dropoff_datetime": "dropoff_datetime",
    },
    "green": {
        "lpep_pickup_datetime": "pickup_datetime",
        "lpep_dropoff_datetime": "dropoff_datetime",
    },
}

OUTPUT_COLUMNS = [
    "taxi_type",
    "vendor_id",
    "pickup_datetime",
    "dropoff_datetime",
    "passenger_count",
    "trip_distance",
    "ratecode_id",
    "store_and_fwd_flag",
    "pickup_location_id",
    "dropoff_location_id",
    "payment_type",
    "fare_amount",
    "extra",
    "mta_tax",
    "tip_amount",
    "tolls_amount",
    "improvement_surcharge",
    "total_amount",
    "congestion_surcharge",
    "airport_fee",
    "ehail_fee",
    "trip_type",
    "source_url",
    "extracted_at",
]

INTEGER_COLUMNS = [
    "vendor_id",
    "pickup_location_id",
    "dropoff_location_id",
    "payment_type",
]

FLOAT_COLUMNS = [
    "passenger_count",
    "trip_distance",
    "ratecode_id",
    "fare_amount",
    "extra",
    "mta_tax",
    "tip_amount",
    "tolls_amount",
    "improvement_surcharge",
    "total_amount",
    "congestion_surcharge",
    "airport_fee",
    "ehail_fee",
    "trip_type",
]

STRING_COLUMNS = ["taxi_type", "store_and_fwd_flag", "source_url"]
DATETIME_COLUMNS = ["pickup_datetime", "dropoff_datetime", "extracted_at"]


def _get_window() -> tuple[datetime, datetime]:
    start_raw = os.environ.get("BRUIN_START_DATETIME") or os.environ.get("BRUIN_START_DATE")
    end_raw = os.environ.get("BRUIN_END_DATETIME") or os.environ.get("BRUIN_END_DATE")
    if not start_raw or not end_raw:
        raise ValueError(
            "BRUIN_START_DATE/BRUIN_START_DATETIME and BRUIN_END_DATE/BRUIN_END_DATETIME are required."
        )

    start = datetime.fromisoformat(start_raw)
    end = datetime.fromisoformat(end_raw)
    if end < start:
        raise ValueError("BRUIN_END_DATE must be greater than or equal to BRUIN_START_DATE.")
    if end == start:
        end = start + timedelta(days=1)
    return start, end


def _get_taxi_types() -> list[str]:
    raw_vars = os.environ.get("BRUIN_VARS")
    if not raw_vars:
        return ["yellow"]

    try:
        vars_dict = json.loads(raw_vars)
    except json.JSONDecodeError as exc:
        raise ValueError("BRUIN_VARS must be valid JSON.") from exc

    taxi_types = vars_dict.get("taxi_types", ["yellow"])
    if not isinstance(taxi_types, list) or not taxi_types:
        raise ValueError("taxi_types must be a non-empty list.")

    parsed_types: list[str] = []
    for taxi_type in taxi_types:
        if not isinstance(taxi_type, str):
            raise ValueError("taxi_types values must be strings.")

        normalized = taxi_type.strip().lower()
        if normalized not in SUPPORTED_TAXI_TYPES:
            supported = ", ".join(sorted(SUPPORTED_TAXI_TYPES))
            raise ValueError(f"Unsupported taxi type '{taxi_type}'. Supported values: {supported}.")
        parsed_types.append(normalized)

    return list(dict.fromkeys(parsed_types))


def _month_starts(start: datetime, end: datetime) -> list[date]:
    month_start = date(start.year, start.month, 1)
    month_starts: list[date] = []
    while month_start < end.date():
        month_starts.append(month_start)
        month_start = month_start + relativedelta(months=1)
    return month_starts


def _download_parquet(url: str) -> pd.DataFrame:
    response = requests.get(url, stream=True, timeout=180)
    if response.status_code in (403, 404):
        print(f"Skipping missing source file: {url}")
        return pd.DataFrame()
    response.raise_for_status()

    with TemporaryDirectory() as temp_dir:
        parquet_path = Path(temp_dir) / "source.parquet"
        with open(parquet_path, "wb") as temp_file:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    temp_file.write(chunk)
        return pd.read_parquet(parquet_path)


def _normalize_frame(frame: pd.DataFrame, taxi_type: str, source_url: str, extracted_at: datetime) -> pd.DataFrame:
    rename_map = {**COMMON_RENAMES, **PICKUP_DROPOFF_RENAMES[taxi_type]}
    frame = frame.rename(columns=rename_map)

    for column in OUTPUT_COLUMNS:
        if column not in frame.columns:
            frame[column] = pd.NA

    frame["taxi_type"] = taxi_type
    frame["source_url"] = source_url
    frame["extracted_at"] = extracted_at
    frame["pickup_datetime"] = pd.to_datetime(frame["pickup_datetime"], errors="coerce")
    frame["dropoff_datetime"] = pd.to_datetime(frame["dropoff_datetime"], errors="coerce")

    return _enforce_output_dtypes(frame[OUTPUT_COLUMNS])


def _enforce_output_dtypes(frame: pd.DataFrame) -> pd.DataFrame:
    for column in INTEGER_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce").astype("Int64")
    for column in FLOAT_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce").astype("Float64")
    for column in STRING_COLUMNS:
        frame[column] = frame[column].astype("string")
    for column in DATETIME_COLUMNS:
        frame[column] = pd.to_datetime(frame[column], errors="coerce")
    return frame


def _empty_output_frame() -> pd.DataFrame:
    frame = pd.DataFrame({column: pd.Series([], dtype="object") for column in OUTPUT_COLUMNS})
    return _enforce_output_dtypes(frame)


def materialize():
    start, end = _get_window()
    taxi_types = _get_taxi_types()
    month_starts = _month_starts(start, end)
    extracted_at = datetime.utcnow()

    frames: list[pd.DataFrame] = []
    for taxi_type in taxi_types:
        for month_start in month_starts:
            if month_start > LAST_AVAILABLE_MONTH:
                print(
                    "Skipping month "
                    f"{month_start:%Y-%m}: TLC trip data for this project is available through 2025-11."
                )
                continue

            file_name = f"{taxi_type}_tripdata_{month_start:%Y-%m}.parquet"
            source_url = f"{BASE_URL}{file_name}"
            source_frame = _download_parquet(source_url)
            if source_frame.empty:
                continue

            source_frame = _normalize_frame(source_frame, taxi_type, source_url, extracted_at)
            source_frame = source_frame[
                (source_frame["pickup_datetime"] >= start)
                & (source_frame["pickup_datetime"] < end)
            ]
            if source_frame.empty:
                continue
            frames.append(source_frame)

    if not frames:
        return _empty_output_frame()

    return _enforce_output_dtypes(pd.concat(frames, ignore_index=True)[OUTPUT_COLUMNS])
