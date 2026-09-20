from dataclasses import dataclass
from datetime import datetime
import json
from uuid import UUID

import psycopg
from psycopg.types.json import Jsonb


@dataclass(frozen=True)
class PredictionLog:
    request_id: UUID
    requested_at: datetime
    model_version: str | None
    features: object
    prediction: float | None
    latency_ms: float
    status_code: int


def request_features(body: bytes) -> object:
    def validate_strings(value):
        if isinstance(value, str):
            value.encode("utf-8")
            if "\x00" in value:
                raise ValueError("Unsupported JSONB text")
        elif isinstance(value, dict):
            for key, item in value.items():
                validate_strings(key)
                validate_strings(item)
        elif isinstance(value, list):
            for item in value:
                validate_strings(item)

    try:
        value = json.loads(body)
        json.dumps(value, allow_nan=False)
        validate_strings(value)
        return value
    except (ValueError, UnicodeError):
        return {"_raw_body_hex": body.hex(), "_encoding": "hex"}


class PostgresRequestLog:
    def __init__(self, database_url: str):
        self.database_url = database_url

    def connect(self):
        return psycopg.connect(
            self.database_url, connect_timeout=3,
            options="-c statement_timeout=3000 -c lock_timeout=3000",
        )

    def initialize(self):
        with self.connect() as connection:
            connection.execute("""
                CREATE TABLE IF NOT EXISTS prediction_requests (
                    request_id UUID PRIMARY KEY,
                    requested_at TIMESTAMPTZ NOT NULL,
                    model_version TEXT,
                    features JSONB NOT NULL,
                    prediction DOUBLE PRECISION,
                    latency_ms DOUBLE PRECISION NOT NULL CHECK (latency_ms >= 0),
                    status_code SMALLINT NOT NULL CHECK (status_code BETWEEN 100 AND 599)
                )
            """)

    def write(self, record: PredictionLog):
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO prediction_requests (
                    request_id, requested_at, model_version, features,
                    prediction, latency_ms, status_code
                ) VALUES (%s, %s, %s, %s, %s, %s, %s)
                """,
                (record.request_id, record.requested_at, record.model_version,
                 Jsonb(record.features), record.prediction, record.latency_ms,
                 record.status_code),
            )
