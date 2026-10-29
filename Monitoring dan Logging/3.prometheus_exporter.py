from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import logging
import math
import os
import re
import signal
import subprocess
import sys
import tempfile
import time
import uuid
from collections.abc import AsyncIterator, Callable, Collection, Iterator, Mapping, Sequence
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from logging.handlers import RotatingFileHandler
from pathlib import Path
from types import FrameType
from typing import Any, TextIO
from wsgiref.simple_server import WSGIServer

import httpx
import psutil
import uvicorn
import yaml
from dotenv import dotenv_values
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    Info,
    ProcessCollector,
    disable_created_metrics,
    generate_latest,
    start_http_server,
)
from prometheus_client.core import CounterMetricFamily, GaugeMetricFamily, Metric
from prometheus_client.registry import Collector
from starlette.requests import ClientDisconnect

# =========
# Constants
# =========

PROJECT_ROOT = Path(__file__).resolve().parent
DOTENV_PATH = PROJECT_ROOT / ".env"

GATEWAY_SERVICE = "inference-gateway"
MODEL_SERVER_SERVICE = "model-server"
GATEWAY_JOB = GATEWAY_SERVICE  
MODEL_SERVER_JOB = MODEL_SERVER_SERVICE

INVOCATIONS_PATH = "/invocations"
PING_PATH = "/ping"
REQUEST_ID_HEADER = "X-Request-ID"
FAULT_HEADER = "X-Fault-Injection"
INJECTED_READ_TIMEOUT_SECONDS = 0.001

_REQUEST_ID_PATTERN = re.compile(r"^[A-Za-z0-9._-]{8,64}$")
_ERROR_CODE_PATTERN = re.compile(r"^[A-Z_]{1,64}$")


# =============
# Configuration
# =============


class ConfigError(ValueError):


def load_environment(dotenv_path: Path | None = DOTENV_PATH) -> dict[str, str]:

    merged: dict[str, str] = {}
    if dotenv_path is not None and dotenv_path.is_file():
        merged.update({key: value for key, value in dotenv_values(dotenv_path).items() if value is not None})
    merged.update(os.environ)
    return merged


class EnvReader:

    _TRUE = frozenset({"1", "true", "yes", "on"})
    _FALSE = frozenset({"0", "false", "no", "off"})

    def __init__(self, env: Mapping[str, str]) -> None:
        self._env = env
        self._errors: list[str] = []

    def _raw(self, name: str) -> str:
        return self._env.get(name, "").strip()

    def text(self, name: str, default: str | None = None) -> str:
        value = self._raw(name)
        if value:
            return value
        if default is None:
            self._errors.append(f"{name} is required but not set.")
            return ""
        return default

    def integer(self, name: str, default: int, *, minimum: int, maximum: int | None = None) -> int:
        raw = self._raw(name)
        if not raw:
            return default
        try:
            value = int(raw)
        except ValueError:
            self._errors.append(f"{name} must be an integer, got {raw!r}.")
            return default
        self._check_range(name, value, minimum, maximum)
        return value

    def number(self, name: str, default: float, *, minimum: float, maximum: float | None = None) -> float:
        raw = self._raw(name)
        if not raw:
            return default
        try:
            value = float(raw)
        except ValueError:
            self._errors.append(f"{name} must be a number, got {raw!r}.")
            return default
        self._check_range(name, value, minimum, maximum)
        return value

    def boolean(self, name: str, default: bool) -> bool:
        raw = self._raw(name).lower()
        if not raw:
            return default
        if raw in self._TRUE:
            return True
        if raw in self._FALSE:
            return False
        self._errors.append(f"{name} must be true/false, got {raw!r}.")
        return default

    def port(self, name: str, default: int) -> int:
        return self.integer(name, default, minimum=1, maximum=65535)

    def path(self, name: str, default: str) -> Path:
        return Path(self.text(name, default)).expanduser()

    def optional_path(self, name: str) -> Path | None:
        raw = self._raw(name)
        return Path(raw).expanduser() if raw else None

    def url(self, name: str, default: str) -> str:
        value = self.text(name, default)
        if not value.startswith(("http://", "https://")):
            self._errors.append(f"{name} must start with http:// or https://, got {value!r}.")
        return value.rstrip("/")

    def choice(self, name: str, default: str, allowed: Collection[str]) -> str:
        value = self.text(name, default).upper()
        if value not in allowed:
            self._errors.append(f"{name} must be one of {sorted(allowed)}, got {value!r}.")
        return value

    def add_error(self, message: str) -> None:
        """Record a cross-field validation problem."""
        self._errors.append(message)

    def raise_if_invalid(self, component: str) -> None:
        if self._errors:
            details = "\n  - ".join(self._errors)
            raise ConfigError(f"Invalid {component} configuration:\n  - {details}")

    def _check_range(self, name: str, value: float, minimum: float, maximum: float | None) -> None:
        if value < minimum or (maximum is not None and value > maximum):
            upper = "" if maximum is None else f" and <= {maximum}"
            self._errors.append(f"{name} must be >= {minimum}{upper}, got {value}.")


@dataclass(frozen=True)
class LogSettings:

    level: str
    directory: Path | None
    max_bytes: int
    backup_count: int

    @classmethod
    def from_reader(cls, reader: EnvReader) -> LogSettings:
        return cls(
            level=reader.choice("LOG_LEVEL", "INFO", {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}),
            directory=reader.optional_path("LOG_DIR"),
            max_bytes=reader.integer("LOG_MAX_BYTES", 10 * 1024 * 1024, minimum=64 * 1024),
            backup_count=reader.integer("LOG_BACKUP_COUNT", 5, minimum=1, maximum=50),
        )

    def file_for(self, service: str) -> Path | None:
        """Rotating log file for ``service``, or ``None`` when file logging is off."""
        return None if self.directory is None else self.directory / f"{service}.jsonl"


@dataclass(frozen=True)
class GatewaySettings:

    host: str
    port: int
    model_server_url: str
    model_artifact_dir: Path
    upstream_timeout_seconds: float
    readiness_interval_seconds: float
    readiness_timeout_seconds: float
    max_records_per_request: int
    max_request_bytes: int
    fault_injection_enabled: bool
    log: LogSettings

    @classmethod
    def from_env(cls, env: Mapping[str, str]) -> GatewaySettings:
        r = EnvReader(env)
        settings = cls(
            host=r.text("GATEWAY_HOST", "127.0.0.1"),
            port=r.port("GATEWAY_PORT", 8000),
            model_server_url=r.url("MODEL_SERVER_URL", "http://127.0.0.1:5001"),
            model_artifact_dir=r.path("MODEL_ARTIFACT_DIR", "./model_artifact"),
            upstream_timeout_seconds=r.number("UPSTREAM_TIMEOUT_SECONDS", 10.0, minimum=0.1, maximum=300),
            readiness_interval_seconds=r.number("READINESS_INTERVAL_SECONDS", 5.0, minimum=1, maximum=300),
            readiness_timeout_seconds=r.number("READINESS_TIMEOUT_SECONDS", 2.0, minimum=0.1, maximum=60),
            max_records_per_request=r.integer("MAX_RECORDS_PER_REQUEST", 1000, minimum=1, maximum=100_000),
            max_request_bytes=r.integer("MAX_REQUEST_BYTES", 8 * 1024 * 1024, minimum=1024),
            fault_injection_enabled=r.boolean("FAULT_INJECTION_ENABLED", False),
            log=LogSettings.from_reader(r),
        )
        r.raise_if_invalid("gateway")
        return settings


@dataclass(frozen=True)
class ModelServerSettings:

    model_artifact_dir: Path
    host: str
    port: int
    workers: int
    request_timeout_seconds: int
    metrics_host: str
    metrics_port: int
    target_name: str
    shutdown_grace_seconds: float
    log: LogSettings

    @classmethod
    def from_env(cls, env: Mapping[str, str]) -> ModelServerSettings:
        r = EnvReader(env)
        settings = cls(
            model_artifact_dir=r.path("MODEL_ARTIFACT_DIR", "./model_artifact"),
            host=r.text("MODEL_SERVER_HOST", "127.0.0.1"),
            port=r.port("MODEL_SERVER_PORT", 5001),
            workers=r.integer("MODEL_SERVER_WORKERS", 1, minimum=1, maximum=32),
            request_timeout_seconds=r.integer("MODEL_SERVER_TIMEOUT_SECONDS", 60, minimum=1, maximum=3600),
            metrics_host=r.text("PROCESS_METRICS_HOST", "127.0.0.1"),
            metrics_port=r.port("PROCESS_METRICS_PORT", 9101),
            target_name=r.text("SERVING_TARGET_NAME", MODEL_SERVER_SERVICE),
            shutdown_grace_seconds=r.number("SHUTDOWN_GRACE_SECONDS", 20.0, minimum=1, maximum=300),
            log=LogSettings.from_reader(r),
        )
        if settings.port == settings.metrics_port:
            r.add_error("MODEL_SERVER_PORT and PROCESS_METRICS_PORT must differ.")
        r.raise_if_invalid("model-server")
        return settings


# =======================
# Structured JSON logging
# =======================

REDACTED = "[REDACTED]"
_MAX_LOG_TEXT = 4000
_SENSITIVE_KEY = re.compile(r"pass(word|wd)?|secret|token|authori[sz]ation|api[_-]?key|cookie|credential", re.I)
_RESERVED_LOG_KEYS = frozenset({"timestamp", "level", "service", "event", "logger", "message", "request_id"})
_request_id: ContextVar[str | None] = ContextVar("request_id", default=None)


@contextlib.contextmanager
def request_context(request_id: str) -> Iterator[None]:

    token = _request_id.set(request_id)
    try:
        yield
    finally:
        _request_id.reset(token)


def redact(fields: Mapping[str, Any]) -> dict[str, Any]:

    clean: dict[str, Any] = {}
    for key, value in fields.items():
        if _SENSITIVE_KEY.search(str(key)):
            clean[key] = REDACTED
        elif isinstance(value, Mapping):
            clean[key] = redact(value)
        else:
            clean[key] = value
    return clean


def _truncate(text: str) -> str:
    return text if len(text) <= _MAX_LOG_TEXT else text[:_MAX_LOG_TEXT] + "...[truncated]"


class JsonFormatter(logging.Formatter):

    def __init__(self, service: str) -> None:
        super().__init__()
        self._service = service

    def format(self, record: logging.LogRecord) -> str:
        entry: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=UTC)
            .isoformat(timespec="milliseconds")
            .replace("+00:00", "Z"),
            "level": record.levelname,
            "service": self._service,
            "event": getattr(record, "event", "log"),
            "logger": record.name,
            "message": _truncate(record.getMessage()),
        }
        request_id = getattr(record, "request_id", None) or _request_id.get()
        if request_id:
            entry["request_id"] = request_id
        fields = getattr(record, "fields", None)
        if isinstance(fields, Mapping):
            entry.update({k: v for k, v in redact(fields).items() if k not in _RESERVED_LOG_KEYS})
        if record.exc_info and record.exc_info[0] is not None:
            exc_type, exc_value, _ = record.exc_info
            entry["exception"] = {
                "type": exc_type.__name__,
                "message": _truncate(str(exc_value)),
                "traceback": _truncate(self.formatException(record.exc_info)),
            }
        return json.dumps(entry, default=str, ensure_ascii=False)


def configure_logging(service: str, settings: LogSettings, *, stream: TextIO | None = None) -> logging.Logger:
 
    formatter = JsonFormatter(service)
    handlers: list[logging.Handler] = [logging.StreamHandler(stream or sys.stdout)]
    log_file = settings.file_for(service)
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(
            RotatingFileHandler(
                log_file, maxBytes=settings.max_bytes, backupCount=settings.backup_count, encoding="utf-8"
            )
        )
    root = logging.getLogger()
    for existing in list(root.handlers):
        root.removeHandler(existing)
        existing.close()
    for handler in handlers:
        handler.setFormatter(formatter)
        root.addHandler(handler)
    root.setLevel(settings.level)
    for noisy in ("httpx", "httpcore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    logging.captureWarnings(True)
    return logging.getLogger(service)


def log_event(
    logger: logging.Logger, level: int, event: str, message: str, *, exc_info: bool = False, **fields: Any
) -> None:

    logger.log(level, message, extra={"event": event, "fields": fields}, exc_info=exc_info)


# ==============
# Model artifact
# ==============

NUMERIC_TYPES = frozenset({"double", "float", "long", "integer"})
INTEGER_TYPES = frozenset({"long", "integer"})
SUPPORTED_TYPES = NUMERIC_TYPES | {"boolean"}


class ModelArtifactError(RuntimeError):

@dataclass(frozen=True)
class FeatureSpec:

    name: str
    type: str
    required: bool = True


@dataclass(frozen=True)
class ModelMetadata:

    artifact_dir: Path
    model_uuid: str
    run_id: str
    mlflow_version: str
    created_utc: str
    flavors: tuple[str, ...]
    features: tuple[FeatureSpec, ...]

    @property
    def feature_names(self) -> tuple[str, ...]:
        return tuple(feature.name for feature in self.features)

    def identity_fields(self) -> dict[str, str]:

        return {"model_uuid": self.model_uuid, "model_run_id": self.run_id}


def load_model_metadata(artifact_dir: Path) -> ModelMetadata:
 
    mlmodel_path = artifact_dir / "MLmodel"
    if not mlmodel_path.is_file():
        raise ModelArtifactError(
            f"No MLmodel file found in '{artifact_dir}'. Set MODEL_ARTIFACT_DIR to the 'model' directory "
            "produced by the Criterion 3 CI run (it contains MLmodel, model.pkl and conda.yaml)."
        )
    try:
        document = yaml.safe_load(mlmodel_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ModelArtifactError(f"Cannot read '{mlmodel_path}': {exc}") from exc
    if not isinstance(document, dict):
        raise ModelArtifactError(f"'{mlmodel_path}' is not a valid MLmodel document.")
    flavors = document.get("flavors") or {}
    if "python_function" not in flavors:
        raise ModelArtifactError("The model has no 'python_function' flavor, so `mlflow models serve` cannot serve it.")
    return ModelMetadata(
        artifact_dir=artifact_dir,
        model_uuid=str(document.get("model_uuid") or "unknown"),
        run_id=str(document.get("run_id") or "unknown"),
        mlflow_version=str(document.get("mlflow_version") or "unknown"),
        created_utc=str(document.get("utc_time_created") or "unknown"),
        flavors=tuple(sorted(flavors)),
        features=_parse_input_schema(document.get("signature")),
    )


def _parse_input_schema(signature: Any) -> tuple[FeatureSpec, ...]:
    if not isinstance(signature, dict) or not signature.get("inputs"):
        raise ModelArtifactError(
            "The model has no input signature. Re-log it with signatures enabled "
            "(mlflow.sklearn.autolog() does this by default)."
        )
    try:
        columns = json.loads(signature["inputs"])
    except (TypeError, json.JSONDecodeError) as exc:
        raise ModelArtifactError("The model signature 'inputs' field is not valid JSON.") from exc
    features: list[FeatureSpec] = []
    for column in columns:
        name, column_type = column.get("name"), column.get("type")
        if not name:
            raise ModelArtifactError("Only column-based (DataFrame) signatures are supported.")
        if column_type not in SUPPORTED_TYPES:
            raise ModelArtifactError(
                f"Unsupported input type {column_type!r} for column {name!r}; supported: {sorted(SUPPORTED_TYPES)}."
            )
        features.append(FeatureSpec(name=name, type=column_type, required=bool(column.get("required", True))))
    if len({feature.name for feature in features}) != len(features):
        raise ModelArtifactError("The model signature contains duplicate column names.")
    return tuple(features)


# ===================
# Input validation
# ===================

SPLIT_FORMAT = "dataframe_split"
RECORDS_FORMAT = "dataframe_records"
_SUPPORTED_FORMATS = (SPLIT_FORMAT, RECORDS_FORMAT)
_MAX_LISTED_NAMES = 5

Value = float | int | bool


class InputValidationError(ValueError):

    def __init__(self, category: str, message: str, *, http_status: int = 400) -> None:
        super().__init__(message)
        self.category = category
        self.http_status = http_status


@dataclass(frozen=True)
class ValidatedBatch:

    columns: tuple[str, ...]
    rows: tuple[tuple[Value, ...], ...]

    @property
    def record_count(self) -> int:
        return len(self.rows)

    def to_dataframe_split(self, exclude: frozenset[str] = frozenset()) -> dict[str, Any]:
        """MLflow ``dataframe_split`` payload, optionally without some columns."""
        keep = [index for index, name in enumerate(self.columns) if name not in exclude]
        return {
            SPLIT_FORMAT: {
                "columns": [self.columns[index] for index in keep],
                "data": [[row[index] for index in keep] for row in self.rows],
            }
        }


def _reject_non_standard_constant(token: str) -> None:
    raise ValueError(f"non-standard JSON constant {token}")


def _decode_json(body: bytes) -> Any:
    try:
        return json.loads(body, parse_constant=_reject_non_standard_constant)
    except (UnicodeDecodeError, ValueError) as exc:
        raise InputValidationError("invalid_json", "Request body is not valid UTF-8 JSON.") from exc


def _format_names(names: Collection[str]) -> str:
    listed = ", ".join(repr(name) for name in sorted(names)[:_MAX_LISTED_NAMES])
    extra = len(names) - _MAX_LISTED_NAMES
    return listed + (f" (+{extra} more)" if extra > 0 else "")


class PayloadValidator:

    def __init__(self, features: Sequence[FeatureSpec], max_records: int) -> None:
        if not features:
            raise ValueError("The feature schema must not be empty.")
        self._features = {feature.name: feature for feature in features}
        self._schema_order = [feature.name for feature in features]
        self._required = {feature.name for feature in features if feature.required}
        self._max_records = max_records

    def validate(self, body: bytes) -> ValidatedBatch:
        """Parse and validate ``body``; raise :class:`InputValidationError` on any problem."""
        document = _decode_json(body)
        if not isinstance(document, dict):
            raise InputValidationError("invalid_structure", "Request body must be a JSON object.")
        present = [key for key in _SUPPORTED_FORMATS if key in document]
        if len(present) != 1:
            raise InputValidationError(
                "invalid_structure", f"Request body must contain exactly one of {list(_SUPPORTED_FORMATS)}."
            )
        if present[0] == SPLIT_FORMAT:
            columns, rows = self._read_split(document[SPLIT_FORMAT])
        else:
            columns, rows = self._read_records(document[RECORDS_FORMAT])
        self._check_record_count(len(rows))
        ordered = self._resolve_column_order(columns)
        return ValidatedBatch(
            columns=tuple(name for name, _ in ordered),
            rows=tuple(self._convert_row(index, row, ordered) for index, row in enumerate(rows)),
        )

    @staticmethod
    def _read_split(section: Any) -> tuple[list[str], list[list[Any]]]:
        if not isinstance(section, dict):
            raise InputValidationError("invalid_structure", f"'{SPLIT_FORMAT}' must be an object.")
        columns, data = section.get("columns"), section.get("data")
        if not isinstance(columns, list) or not all(isinstance(column, str) for column in columns):
            raise InputValidationError("invalid_structure", f"'{SPLIT_FORMAT}.columns' must be a list of strings.")
        if not isinstance(data, list):
            raise InputValidationError("invalid_structure", f"'{SPLIT_FORMAT}.data' must be a list of rows.")
        for index, row in enumerate(data):
            if not isinstance(row, list) or len(row) != len(columns):
                raise InputValidationError(
                    "invalid_structure",
                    f"Row {index} must be a list with exactly {len(columns)} values (one per column).",
                )
        return columns, data

    @staticmethod
    def _read_records(section: Any) -> tuple[list[str], list[list[Any]]]:
        if not isinstance(section, list) or not all(isinstance(record, dict) for record in section):
            raise InputValidationError("invalid_structure", f"'{RECORDS_FORMAT}' must be a list of objects.")
        if not section:
            return [], []
        columns = list(section[0].keys())
        for index, record in enumerate(section):
            if set(record.keys()) != set(columns):
                raise InputValidationError(
                    "invalid_structure", f"Record {index} does not have the same fields as record 0."
                )
        return columns, [[record[name] for name in columns] for record in section]

    def _check_record_count(self, count: int) -> None:
        if count == 0:
            raise InputValidationError("empty_batch", "The request contains no records.")
        if count > self._max_records:
            raise InputValidationError(
                "too_many_records",
                f"The request contains {count} records; the limit is {self._max_records}.",
                http_status=413,
            )

    def _resolve_column_order(self, columns: Sequence[str]) -> list[tuple[str, int]]:
   
        if len(columns) != len(set(columns)):
            duplicates = {name for name in columns if columns.count(name) > 1}
            raise InputValidationError("duplicate_columns", f"Duplicate columns: {_format_names(duplicates)}.")
        provided = set(columns)
        unexpected = provided - self._features.keys()
        if unexpected:
            raise InputValidationError("unexpected_columns", f"Unknown columns: {_format_names(unexpected)}.")
        missing = self._required - provided
        if missing:
            raise InputValidationError("missing_columns", f"Missing required columns: {_format_names(missing)}.")
        position = {name: index for index, name in enumerate(columns)}
        return [(name, position[name]) for name in self._schema_order if name in provided]

    def _convert_row(self, row_index: int, row: Sequence[Any], ordered: Sequence[tuple[str, int]]) -> tuple[Value, ...]:
        return tuple(
            self._convert_value(row_index, name, row[position], self._features[name].type) for name, position in ordered
        )

    @staticmethod
    def _convert_value(row_index: int, name: str, value: Any, column_type: str) -> Value:
        if column_type == "boolean":
            if isinstance(value, bool):
                return value
            raise InputValidationError("invalid_value", f"Row {row_index}: field {name!r} must be a boolean.")
        is_number = isinstance(value, (int, float)) and not isinstance(value, bool)
        if column_type in NUMERIC_TYPES and is_number and math.isfinite(value):
            if column_type not in INTEGER_TYPES:
                return float(value)
            if float(value).is_integer():
                return int(value)
            raise InputValidationError("invalid_value", f"Row {row_index}: field {name!r} must be an integer.")
        raise InputValidationError("invalid_value", f"Row {row_index}: field {name!r} must be a finite number.")


# ==================
# Prometheus metrics
# ==================


class Outcome(StrEnum):

    SUCCESS = "success"
    INVALID_INPUT = "invalid_input"  # client error: rejected before reaching the model
    MODEL_ERROR = "model_error"  # model server answered with an error or an unusable response
    TIMEOUT = "timeout"  # no answer within UPSTREAM_TIMEOUT_SECONDS
    UNAVAILABLE = "unavailable"  # model server unreachable (connection refused/reset)
    GATEWAY_ERROR = "gateway_error"  # unexpected defect inside the gateway itself


SERVER_SIDE_FAILURES: frozenset[Outcome] = frozenset(
    {Outcome.MODEL_ERROR, Outcome.TIMEOUT, Outcome.UNAVAILABLE, Outcome.GATEWAY_ERROR}
)

LATENCY_BUCKETS_SECONDS = (
    0.005,
    0.01,
    0.025,
    0.05,
    0.075,
    0.1,
    0.25,
    0.5,
    0.75,
    1.0,
    1.5,
    2.5,
    5.0,
    7.5,
    10.0,
    15.0,
    30.0,
)
RECORD_BUCKETS = (1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000)
REQUEST_BYTES_BUCKETS = tuple(512 * 2**i for i in range(15))  # 512 B .. 8 MiB
RESPONSE_BYTES_BUCKETS = tuple(32 * 2**i for i in range(10))  # 32 B .. 16 KiB


@dataclass(frozen=True)
class InferenceMetrics:

    requests: Counter
    duration: Histogram
    in_flight: Gauge
    input_records: Histogram
    request_bytes: Histogram
    response_bytes: Histogram
    model_ready: Gauge
    fault_injections: Counter
    model_info: Info

    def observe_completion(
        self,
        *,
        outcome: Outcome,
        duration_seconds: float,
        request_bytes: int,
        response_bytes: int,
        record_count: int | None,
    ) -> None:

        self.requests.labels(outcome=outcome.value).inc()
        self.duration.labels(outcome=outcome.value).observe(duration_seconds)
        self.request_bytes.observe(request_bytes)
        self.response_bytes.observe(response_bytes)
        if record_count is not None:
            self.input_records.observe(record_count)


def create_inference_metrics(registry: CollectorRegistry, fault_modes: Collection[str]) -> InferenceMetrics:

    metrics = InferenceMetrics(
        requests=Counter(
            "inference_requests",
            "Completed inference requests handled by the gateway, by outcome.",
            ["outcome"],
            registry=registry,
        ),
        duration=Histogram(
            "inference_duration_seconds",
            "Gateway handler time per inference request: from handler entry to response ready "
            "(body read + validation + model-server call + serialisation).",
            ["outcome"],
            buckets=LATENCY_BUCKETS_SECONDS,
            registry=registry,
        ),
        in_flight=Gauge(
            "inference_in_flight", "Inference requests currently being processed by the gateway.", registry=registry
        ),
        input_records=Histogram(
            "inference_input_records",
            "Number of input records (rows) per schema-valid inference request.",
            buckets=RECORD_BUCKETS,
            registry=registry,
        ),
        request_bytes=Histogram(
            "inference_request_bytes",
            "Size in bytes of the inference request body received by the gateway.",
            buckets=REQUEST_BYTES_BUCKETS,
            registry=registry,
        ),
        response_bytes=Histogram(
            "inference_response_bytes",
            "Size in bytes of the inference response body returned by the gateway.",
            buckets=RESPONSE_BYTES_BUCKETS,
            registry=registry,
        ),
        model_ready=Gauge(
            "model_ready",
            "1 when the model server's /ping health check last succeeded, otherwise 0.",
            registry=registry,
        ),
        fault_injections=Counter(
            "inference_fault_injections",
            "LOCAL TESTING ONLY: requests deliberately perturbed via the X-Fault-Injection header.",
            ["mode"],
            registry=registry,
        ),
        model_info=Info(
            "inference_model", "Identity of the MLflow model artifact served behind the gateway.", registry=registry
        ),
    )
    for outcome in Outcome:
        metrics.requests.labels(outcome=outcome.value)
        metrics.duration.labels(outcome=outcome.value)
    for mode in fault_modes:
        metrics.fault_injections.labels(mode=mode)
    metrics.model_ready.set(0)
    return metrics


def register_gateway_process_metrics(registry: CollectorRegistry) -> None:

    ProcessCollector(namespace="gateway", registry=registry)


# =====================
# Model-server resource 
# =====================

_PROCESS_GONE = (psutil.NoSuchProcess, psutil.ZombieProcess)


class MisattributionError(RuntimeError):


@dataclass(frozen=True)
class ProcessTreeSnapshot:

    root_pid: int
    process_count: int
    cpu_seconds: float
    resident_memory_bytes: int
    uptime_seconds: float


class ProcessTreeSampler:

    def __init__(
        self,
        root_pid: Callable[[], int | None],
        *,
        clock: Callable[[], float] = time.time,
        forbidden_pids: Collection[int] | None = None,
    ) -> None:
        self._root_pid = root_pid
        self._clock = clock
        self._forbidden = frozenset(forbidden_pids if forbidden_pids is not None else {os.getpid()})

    def snapshot(self) -> ProcessTreeSnapshot | None:
   
        pid = self._root_pid()
        if pid is None:
            return None
        try:
            root = psutil.Process(pid)
            started_at = root.create_time()
            tree = [root, *root.children(recursive=True)]
        except (*_PROCESS_GONE, psutil.AccessDenied):
            return None
        if self._forbidden & {process.pid for process in tree}:
            raise MisattributionError(
                f"Refusing to report process tree rooted at PID {pid}: it contains the exporter itself."
            )
        cpu_seconds, rss_bytes, alive = 0.0, 0, 0
        for process in tree:
            try:
                with process.oneshot():
                    times = process.cpu_times()
                    rss = process.memory_info().rss
            except _PROCESS_GONE:
                continue  # exited between listing and sampling; its CPU moves to the parent's children_* time
            cpu_seconds += times.user + times.system
            cpu_seconds += getattr(times, "children_user", 0.0) + getattr(times, "children_system", 0.0)
            rss_bytes += rss
            alive += 1
        if alive == 0:
            return None
        return ProcessTreeSnapshot(pid, alive, cpu_seconds, rss_bytes, max(0.0, self._clock() - started_at))


class ServingProcessCollector(Collector):

    def __init__(self, sampler: ProcessTreeSampler, target: str, logger: logging.Logger) -> None:
        self._sampler = sampler
        self._target = target
        self._logger = logger
        self._last_observed: bool | None = None

    def describe(self) -> Iterator[Metric]:
        yield from self._families()

    def collect(self) -> Iterator[Metric]:
        try:
            snapshot = self._sampler.snapshot()
        except MisattributionError as exc:
            log_event(self._logger, logging.ERROR, "process_metrics_misattribution", str(exc))
            snapshot = None
        self._log_transition(snapshot)
        observed, cpu, memory, uptime, count = self._families()
        observed.add_metric([self._target], 1.0 if snapshot else 0.0)
        yield observed
        if snapshot is None:
            return  # absent series are more honest than zeros for an unobservable process
        cpu.add_metric([self._target], snapshot.cpu_seconds)
        memory.add_metric([self._target], snapshot.resident_memory_bytes)
        uptime.add_metric([self._target], snapshot.uptime_seconds)
        count.add_metric([self._target], snapshot.process_count)
        yield from (cpu, memory, uptime, count)

    @staticmethod
    def _families() -> tuple[
        GaugeMetricFamily, CounterMetricFamily, GaugeMetricFamily, GaugeMetricFamily, GaugeMetricFamily
    ]:
        labels = ["target"]
        return (
            GaugeMetricFamily(
                "serving_process_observed",
                "1 when the model-serving process tree could be sampled, otherwise 0.",
                labels=labels,
            ),
            CounterMetricFamily(
                "serving_process_cpu_seconds",
                "Cumulative user+system CPU seconds of the model-serving process tree.",
                labels=labels,
            ),
            GaugeMetricFamily(
                "serving_process_resident_memory_bytes",
                "Resident set size summed over the model-serving process tree (bytes).",
                labels=labels,
            ),
            GaugeMetricFamily(
                "serving_process_uptime_seconds", "Seconds since the root model-serving process started.", labels=labels
            ),
            GaugeMetricFamily(
                "serving_process_count", "Number of live processes in the model-serving process tree.", labels=labels
            ),
        )

    def _log_transition(self, snapshot: ProcessTreeSnapshot | None) -> None:
        observed = snapshot is not None
        if observed == self._last_observed:
            return
        self._last_observed = observed
        if snapshot is not None:
            log_event(
                self._logger,
                logging.INFO,
                "serving_process_observed",
                "Model-serving process tree is observable.",
                target=self._target,
                root_pid=snapshot.root_pid,
                process_count=snapshot.process_count,
            )
        else:
            log_event(
                self._logger,
                logging.WARNING,
                "serving_process_unobserved",
                "Model-serving process tree cannot be observed.",
                target=self._target,
            )


# ========================
# Model-server HTTP client
# ========================


class UpstreamError(Exception):

    def __init__(self, outcome: Outcome, category: str, http_status: int, detail: str) -> None:
        super().__init__(detail)
        self.outcome = outcome
        self.category = category
        self.http_status = http_status
        self.detail = detail


def _error_code(response: httpx.Response) -> str:
    try:
        code = response.json().get("error_code", "")
    except (ValueError, AttributeError):
        return "no_error_code"
    return code if isinstance(code, str) and _ERROR_CODE_PATTERN.match(code) else "no_error_code"


def _check_predictions(body: bytes, expected_records: int) -> None:
    try:
        document = json.loads(body)
    except ValueError as exc:
        raise UpstreamError(
            Outcome.MODEL_ERROR, "invalid_upstream_response", 502, "model server returned non-JSON content"
        ) from exc
    predictions = document.get("predictions") if isinstance(document, dict) else None
    if not isinstance(predictions, list) or len(predictions) != expected_records:
        raise UpstreamError(
            Outcome.MODEL_ERROR,
            "invalid_upstream_response",
            502,
            f"expected {expected_records} predictions in the response",
        )


class ModelServerClient:

    def __init__(self, http: httpx.AsyncClient, *, inference_timeout: float, readiness_timeout: float) -> None:
        self._http = http
        self._inference_timeout = inference_timeout
        self._readiness_timeout = readiness_timeout

    async def predict(
        self,
        payload: Mapping[str, Any],
        *,
        request_id: str,
        expected_records: int,
        read_timeout_override: float | None = None,
    ) -> bytes:

        timeout = httpx.Timeout(self._inference_timeout)
        if read_timeout_override is not None:
            timeout = httpx.Timeout(self._inference_timeout, read=read_timeout_override)
        try:
            response = await self._http.post(
                INVOCATIONS_PATH, json=payload, headers={REQUEST_ID_HEADER: request_id}, timeout=timeout
            )
        except httpx.TimeoutException as exc:
            raise UpstreamError(
                Outcome.TIMEOUT, "upstream_timeout", 504, f"{type(exc).__name__}: no response within the timeout"
            ) from exc
        except httpx.TransportError as exc:
            raise UpstreamError(
                Outcome.UNAVAILABLE, "upstream_unreachable", 503, f"{type(exc).__name__}: {exc}"
            ) from exc
        status = response.status_code
        if status >= 500:
            raise UpstreamError(
                Outcome.MODEL_ERROR, "upstream_server_error", 502, f"HTTP {status} ({_error_code(response)})"
            )
        if status >= 400:
            # The gateway already validated the payload, so a rejection is a serving-path contract failure.
            raise UpstreamError(
                Outcome.MODEL_ERROR, "upstream_rejected", 502, f"HTTP {status} ({_error_code(response)})"
            )
        if status != 200:
            raise UpstreamError(Outcome.MODEL_ERROR, "upstream_unexpected_status", 502, f"HTTP {status}")
        _check_predictions(response.content, expected_records)
        return response.content

    async def is_ready(self) -> bool:
    
        try:
            response = await self._http.get(PING_PATH, timeout=self._readiness_timeout)
        except httpx.TransportError:  # includes timeouts
            return False
        return response.status_code == 200


# ===================
# Inference (FastAPI)
# ===================


class FaultMode(StrEnum):

    SCHEMA_VIOLATION = "schema-violation"  # forward the batch without its last column -> MLflow rejects it
    UPSTREAM_TIMEOUT = "upstream-timeout"  # forward with a 1 ms read timeout -> genuine client-side timeout


@dataclass(frozen=True)
class InferenceResult:

    outcome: Outcome
    status_code: int
    body: bytes
    record_count: int | None = None
    error_category: str | None = None
    error_detail: str | None = None
    fault_mode: FaultMode | None = None


class PayloadTooLargeError(InputValidationError):

    def __init__(self, limit: int, bytes_received: int) -> None:
        super().__init__("payload_too_large", f"Request body exceeds {limit} bytes.", http_status=413)
        self.bytes_received = bytes_received


def resolve_request_id(candidate: str | None) -> str:

    return candidate if candidate and _REQUEST_ID_PATTERN.fullmatch(candidate) else uuid.uuid4().hex


async def read_limited_body(request: Request, limit: int) -> bytes:
   
    declared = request.headers.get("content-length", "")
    if declared.isdigit() and int(declared) > limit:
        raise PayloadTooLargeError(limit, 0)
    chunks: list[bytes] = []
    size = 0
    async for chunk in request.stream():
        size += len(chunk)
        if size > limit:
            raise PayloadTooLargeError(limit, size)
        chunks.append(chunk)
    return b"".join(chunks)


def _require_json_content_type(content_type: str | None) -> None:
    if (content_type or "").split(";", 1)[0].strip().lower() != "application/json":
        raise InputValidationError("unsupported_media_type", "Content-Type must be application/json.", http_status=415)


def _error_body(category: str, message: str, request_id: str) -> bytes:
    return json.dumps({"error": {"category": category, "message": message, "request_id": request_id}}).encode()


class InferenceService:

    def __init__(
        self,
        *,
        settings: GatewaySettings,
        metadata: ModelMetadata,
        validator: PayloadValidator,
        client: ModelServerClient,
        metrics: InferenceMetrics,
        logger: logging.Logger,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self._settings = settings
        self._metadata = metadata
        self._validator = validator
        self._client = client
        self._metrics = metrics
        self._logger = logger
        self._clock = clock

    async def handle(self, request: Request, request_id: str) -> InferenceResult:
        
        started = self._clock()
        request_bytes = 0
        with self._metrics.in_flight.track_inprogress():
            try:
                body = await read_limited_body(request, self._settings.max_request_bytes)
                request_bytes = len(body)
                result = await self._process(body, request.headers, request_id)
            except PayloadTooLargeError as exc:
                request_bytes = exc.bytes_received
                result = self._rejected(exc, request_id)
            except ClientDisconnect:
                disconnected = InputValidationError(
                    "client_disconnected", "Client disconnected before sending the full body."
                )
                result = self._rejected(disconnected, request_id)
            except asyncio.CancelledError:
                log_event(self._logger, logging.WARNING, "inference_cancelled", "Inference request cancelled.")
                raise
            except Exception as exc:
                log_event(
                    self._logger,
                    logging.ERROR,
                    "gateway_internal_error",
                    "Unexpected gateway error while handling an inference request.",
                    exc_info=True,
                )
                result = InferenceResult(
                    outcome=Outcome.GATEWAY_ERROR,
                    status_code=500,
                    body=_error_body("gateway_internal_error", "Internal gateway error.", request_id),
                    error_category="gateway_internal_error",
                    error_detail=type(exc).__name__,
                )
            duration = self._clock() - started
            self._metrics.observe_completion(
                outcome=result.outcome,
                duration_seconds=duration,
                request_bytes=request_bytes,
                response_bytes=len(result.body),
                record_count=result.record_count,
            )
        self._log_completion(result, duration, request_bytes)
        return result

    async def _process(self, body: bytes, headers: Mapping[str, str], request_id: str) -> InferenceResult:
        try:
            _require_json_content_type(headers.get("content-type"))
            fault = self._resolve_fault(headers.get(FAULT_HEADER))
            batch = self._validator.validate(body)
        except InputValidationError as exc:
            return self._rejected(exc, request_id)

        exclude = frozenset({batch.columns[-1]}) if fault is FaultMode.SCHEMA_VIOLATION else frozenset()
        read_override = INJECTED_READ_TIMEOUT_SECONDS if fault is FaultMode.UPSTREAM_TIMEOUT else None
        if fault is not None:
            self._metrics.fault_injections.labels(mode=fault.value).inc()
        try:
            prediction = await self._client.predict(
                batch.to_dataframe_split(exclude=exclude),
                request_id=request_id,
                expected_records=batch.record_count,
                read_timeout_override=read_override,
            )
        except UpstreamError as failure:
            return InferenceResult(
                outcome=failure.outcome,
                status_code=failure.http_status,
                body=_error_body(failure.category, "The model server could not produce a prediction.", request_id),
                record_count=batch.record_count,
                error_category=failure.category,
                error_detail=failure.detail,
                fault_mode=fault,
            )
        return InferenceResult(Outcome.SUCCESS, 200, prediction, record_count=batch.record_count, fault_mode=fault)

    def _resolve_fault(self, header_value: str | None) -> FaultMode | None:
        if header_value is None:
            return None
        if not self._settings.fault_injection_enabled:
            log_event(
                self._logger,
                logging.WARNING,
                "fault_injection_ignored",
                f"{FAULT_HEADER} header ignored because FAULT_INJECTION_ENABLED is false.",
            )
            return None
        try:
            return FaultMode(header_value.strip().lower())
        except ValueError as exc:
            allowed = ", ".join(mode.value for mode in FaultMode)
            raise InputValidationError(
                "unsupported_fault_mode", f"Unsupported {FAULT_HEADER} value; expected one of: {allowed}."
            ) from exc

    @staticmethod
    def _rejected(exc: InputValidationError, request_id: str) -> InferenceResult:
        return InferenceResult(
            outcome=Outcome.INVALID_INPUT,
            status_code=exc.http_status,
            body=_error_body(exc.category, str(exc), request_id),
            error_category=exc.category,
            error_detail=str(exc),
        )

    def _log_completion(self, result: InferenceResult, duration: float, request_bytes: int) -> None:
        if result.outcome is Outcome.SUCCESS:
            level = logging.INFO
        elif result.outcome in SERVER_SIDE_FAILURES:
            level = logging.ERROR
        else:
            level = logging.WARNING
        fields: dict[str, Any] = {
            "outcome": result.outcome.value,
            "http_status": result.status_code,
            "duration_ms": round(duration * 1000, 3),
            "records": result.record_count,
            "request_bytes": request_bytes,
            "response_bytes": len(result.body),
            **self._metadata.identity_fields(),
        }
        if result.error_category:
            fields.update(error_category=result.error_category, error_detail=result.error_detail)
        if result.fault_mode:
            fields["fault_injection"] = result.fault_mode.value
        log_event(self._logger, level, "inference_completed", f"Inference request {result.outcome.value}.", **fields)


class ReadinessMonitor:

    def __init__(
        self, client: ModelServerClient, metrics: InferenceMetrics, interval: float, logger: logging.Logger
    ) -> None:
        self._client = client
        self._metrics = metrics
        self._interval = interval
        self._logger = logger
        self._ready: bool | None = None

    @property
    def ready(self) -> bool:
        return bool(self._ready)

    async def check_once(self) -> bool:
        ready = await self._client.is_ready()
        self._metrics.model_ready.set(1 if ready else 0)
        if ready != self._ready:
            log_event(
                self._logger,
                logging.INFO if ready else logging.WARNING,
                "model_readiness_changed",
                f"Model server is {'ready' if ready else 'NOT ready'}.",
                previous="unknown" if self._ready is None else ("ready" if self._ready else "not_ready"),
                current="ready" if ready else "not_ready",
            )
        self._ready = ready
        return ready

    async def run(self) -> None:
        while True:
            await self.check_once()
            await asyncio.sleep(self._interval)

    def mark_unknown(self) -> None:
        self._ready = None
        self._metrics.model_ready.set(0)


def _on_monitor_exit(task: asyncio.Task[None], monitor: ReadinessMonitor, logger: logging.Logger) -> None:
    if task.cancelled() or task.exception() is None:
        return
    monitor.mark_unknown()
    log_event(
        logger,
        logging.CRITICAL,
        "readiness_monitor_crashed",
        "Readiness monitor stopped unexpectedly; model_ready forced to 0.",
        error_type=type(task.exception()).__name__,
    )


def create_app(
    settings: GatewaySettings,
    metadata: ModelMetadata,
    logger: logging.Logger,
    *,
    registry: CollectorRegistry | None = None,
    transport: httpx.AsyncBaseTransport | None = None,
    clock: Callable[[], float] = time.perf_counter,
) -> FastAPI:

    registry = registry or CollectorRegistry()
    metrics = create_inference_metrics(registry, fault_modes=[mode.value for mode in FaultMode])
    register_gateway_process_metrics(registry)
    metrics.model_info.info(
        {"model_uuid": metadata.model_uuid, "run_id": metadata.run_id, "mlflow_version": metadata.mlflow_version}
    )
    http = httpx.AsyncClient(
        base_url=settings.model_server_url,
        transport=transport,
        limits=httpx.Limits(max_connections=100, max_keepalive_connections=20),
    )
    client = ModelServerClient(
        http, inference_timeout=settings.upstream_timeout_seconds, readiness_timeout=settings.readiness_timeout_seconds
    )
    service = InferenceService(
        settings=settings,
        metadata=metadata,
        validator=PayloadValidator(metadata.features, settings.max_records_per_request),
        client=client,
        metrics=metrics,
        logger=logger,
        clock=clock,
    )
    monitor = ReadinessMonitor(client, metrics, settings.readiness_interval_seconds, logger)

    @contextlib.asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        task = asyncio.create_task(monitor.run(), name="readiness-monitor")
        task.add_done_callback(lambda finished: _on_monitor_exit(finished, monitor, logger))
        log_event(
            logger,
            logging.INFO,
            "gateway_started",
            "Inference gateway started.",
            listen=f"{settings.host}:{settings.port}",
            model_server_url=settings.model_server_url,
            feature_count=len(metadata.features),
            fault_injection_enabled=settings.fault_injection_enabled,
            **metadata.identity_fields(),
        )
        try:
            yield
        finally:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
            monitor.mark_unknown()
            await http.aclose()
            log_event(logger, logging.INFO, "gateway_stopped", "Inference gateway stopped.")

    app = FastAPI(
        title="Credit-scoring inference gateway", lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None
    )
    app.state.metrics = metrics
    app.state.monitor = monitor
    app.state.registry = registry

    @app.post(INVOCATIONS_PATH)
    async def invocations(request: Request) -> Response:
        request_id = resolve_request_id(request.headers.get(REQUEST_ID_HEADER))
        with request_context(request_id):
            result = await service.handle(request, request_id)
        return Response(
            content=result.body,
            status_code=result.status_code,
            media_type="application/json",
            headers={REQUEST_ID_HEADER: request_id},
        )

    @app.get("/metrics")
    async def metrics_endpoint() -> Response:
        return Response(content=generate_latest(registry), media_type=CONTENT_TYPE_LATEST)

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok", "service": GATEWAY_SERVICE}

    @app.get("/readyz")
    async def readyz() -> JSONResponse:
        return JSONResponse({"model_ready": monitor.ready}, status_code=200 if monitor.ready else 503)

    return app


# =======================
# Model-server supervisor
# =======================

_MAX_OUTPUT_LINE = 2000
_KILL_WAIT_SECONDS = 5.0


def build_serve_command(settings: ModelServerSettings) -> list[str]:

    return [
        sys.executable, "-m", "mlflow", "models", "serve",
        "--model-uri", str(settings.model_artifact_dir.resolve()),
        "--host", settings.host,
        "--port", str(settings.port),
        "--workers", str(settings.workers),
        "--timeout", str(settings.request_timeout_seconds),
        "--env-manager", "local",
    ]  # fmt: skip


def build_serve_environment(workdir: Path, base: Mapping[str, str] | None = None) -> dict[str, str]:

    env = dict(os.environ if base is None else base)
    env["MLFLOW_TRACKING_URI"] = (workdir / "mlruns").as_uri()
    return env


def _new_session_kwargs() -> dict[str, Any]:

    if os.name == "posix":
        return {"start_new_session": True}
    return {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}


class ModelServerSupervisor:

    def __init__(
        self,
        settings: ModelServerSettings,
        logger: logging.Logger,
        *,
        popen: Callable[..., subprocess.Popen[str]] = subprocess.Popen,
    ) -> None:
        self._settings = settings
        self._logger = logger
        self._popen = popen
        self._process: subprocess.Popen[str] | None = None
        self._shutdown_requested = False

    def child_pid(self) -> int | None:

        if self._process is None or self._process.poll() is not None:
            return None
        return self._process.pid

    def run(self) -> int:
   
        metrics_server = self._start_metrics_server()
        self._install_signal_handlers()
        try:
            with tempfile.TemporaryDirectory(prefix="mlflow-serve-", ignore_cleanup_errors=True) as workdir:
                return self._serve(Path(workdir))
        finally:
            metrics_server.shutdown()
            metrics_server.server_close()
            log_event(self._logger, logging.INFO, "model_server_supervisor_stopped", "Supervisor stopped.")

    def _start_metrics_server(self) -> WSGIServer:
        registry = CollectorRegistry()
        sampler = ProcessTreeSampler(self.child_pid)
        registry.register(ServingProcessCollector(sampler, self._settings.target_name, self._logger))
        server, _ = start_http_server(self._settings.metrics_port, addr=self._settings.metrics_host, registry=registry)
        return server

    def _serve(self, workdir: Path) -> int:
        command = build_serve_command(self._settings)
        try:
            self._process = self._popen(
                command,
                cwd=workdir,
                env=build_serve_environment(workdir),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                **_new_session_kwargs(),
            )
            log_event(
                self._logger,
                logging.INFO,
                "model_server_started",
                "MLflow scoring server started.",
                pid=self._process.pid,
                command=" ".join(command),
                workers=self._settings.workers,
                serve_address=f"{self._settings.host}:{self._settings.port}",
                metrics_address=f"{self._settings.metrics_host}:{self._settings.metrics_port}",
            )
            self._relay_output()
            return_code = self._process.wait()
            expected = self._shutdown_requested
            log_event(
                self._logger,
                logging.INFO if expected else logging.ERROR,
                "model_server_exited",
                "MLflow scoring server exited" + (" after a shutdown request." if expected else " unexpectedly."),
                return_code=return_code,
            )
            return 0 if expected else (return_code or 1)
        finally:
            self._ensure_stopped()

    def _relay_output(self) -> None:

        assert self._process is not None and self._process.stdout is not None
        for line in self._process.stdout:
            text = line.rstrip()
            if text:
                log_event(self._logger, logging.INFO, "model_server_output", text[:_MAX_OUTPUT_LINE])

    def _install_signal_handlers(self) -> None:
        for signum in (signal.SIGINT, signal.SIGTERM):
            signal.signal(signum, self._request_shutdown)

    def _request_shutdown(self, signum: int, _frame: FrameType | None) -> None:
        if self._shutdown_requested:
            return  # e.g. Ctrl+C reaches the group, then the parent of `all` mode sends SIGTERM
        self._shutdown_requested = True
        log_event(
            self._logger,
            logging.INFO,
            "model_server_shutdown_requested",
            "Shutdown requested; terminating the MLflow scoring server.",
            signal=signal.Signals(signum).name,
        )
        self._signal_tree(kill=False)  # non-blocking; the output relay ends when the tree exits

    def _signal_tree(self, *, kill: bool) -> None:
        if self._process is None:
            return
        if os.name == "posix":
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.killpg(self._process.pid, signal.SIGKILL if kill else signal.SIGTERM)
            return
        with contextlib.suppress(psutil.NoSuchProcess):
            root = psutil.Process(self._process.pid)
            for process in [*root.children(recursive=True), root]:
                with contextlib.suppress(psutil.NoSuchProcess):
                    process.kill() if kill else process.terminate()

    def _ensure_stopped(self) -> None:
        if self._process is None:
            return
        if self._process.poll() is not None:
            self._signal_tree(kill=False)  # reap workers orphaned in the process group, if any
            return
        self._signal_tree(kill=False)
        try:
            self._process.wait(timeout=self._settings.shutdown_grace_seconds)
        except subprocess.TimeoutExpired:
            log_event(self._logger, logging.WARNING, "model_server_kill", "Grace period expired; killing the tree.")
            self._signal_tree(kill=True)
            self._process.wait(timeout=_KILL_WAIT_SECONDS)


# ======================
# Command-line interface
# ======================


def run_gateway(env: Mapping[str, str]) -> int:

    try:
        settings = GatewaySettings.from_env(env)
    except ConfigError as exc:
        print(exc, file=sys.stderr)
        return 2
    logger = configure_logging(GATEWAY_SERVICE, settings.log)
    try:
        metadata = load_model_metadata(settings.model_artifact_dir)
    except ModelArtifactError as exc:
        log_event(logger, logging.CRITICAL, "model_artifact_invalid", str(exc))
        return 2
    disable_created_metrics()  # drop redundant *_created series (process-wide client setting)
    app = create_app(settings, metadata, logger)
    uvicorn.run(
        app, host=settings.host, port=settings.port, log_config=None, access_log=False, timeout_graceful_shutdown=10
    )
    return 0


def run_model_server(env: Mapping[str, str]) -> int:

    try:
        settings = ModelServerSettings.from_env(env)
    except ConfigError as exc:
        print(exc, file=sys.stderr)
        return 2
    logger = configure_logging(MODEL_SERVER_SERVICE, settings.log)
    try:
        metadata = load_model_metadata(settings.model_artifact_dir)
    except ModelArtifactError as exc:
        log_event(logger, logging.CRITICAL, "model_artifact_invalid", str(exc))
        return 2
    log_event(
        logger,
        logging.INFO,
        "model_artifact_loaded",
        "Model artifact metadata loaded.",
        artifact_dir=str(settings.model_artifact_dir),
        mlflow_version=metadata.mlflow_version,
        flavors=list(metadata.flavors),
        feature_count=len(metadata.features),
        **metadata.identity_fields(),
    )
    return ModelServerSupervisor(settings, logger).run()


def run_all(env: Mapping[str, str]) -> int:

    try:
        GatewaySettings.from_env(env)
        ModelServerSettings.from_env(env)
    except ConfigError as exc:
        print(exc, file=sys.stderr)
        return 2
    child = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "model-server"], env={**os.environ, **env})
    try:
        return run_gateway(env)
    finally:
        if child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()


COMPONENTS: dict[str, tuple[Callable[[Mapping[str, str]], int], str]] = {
    "all": (run_all, "Both components for a local run: model-server (child process) + gateway (default)."),
    "model-server": (run_model_server, "Run `mlflow models serve` on the CI artifact; export process metrics (:9101)."),
    "gateway": (run_gateway, "Inference gateway + exporter of inference_* and model_ready metrics (:8000)."),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="3.prometheus_exporter.py",
        description="Serving and Prometheus exporters for the MLflow credit-scoring model (Kriteria 4).",
    )
    subcommands = parser.add_subparsers(dest="component", metavar="{all,model-server,gateway}")
    for name, (_, help_text) in COMPONENTS.items():
        subcommands.add_parser(name, help=help_text, description=help_text)
    parser.set_defaults(component="all")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    runner, _ = COMPONENTS[args.component]
    return runner(load_environment())


if __name__ == "__main__":
    raise SystemExit(main())
