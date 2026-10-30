from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import random
import sys
import threading
import time
import uuid
from collections import Counter
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

SERVICE_NAME = "inference-client"
DEFAULT_URL = "http://127.0.0.1:8000/invocations"
DEFAULT_DATA = Path(__file__).resolve().parent / "data" / "test.csv"
TARGET_COLUMN = "credit_risk"
REQUEST_ID_HEADER = "X-Request-ID"
FAULT_HEADER = "X-Fault-Injection"
FAULT_MODES = ("schema-violation", "upstream-timeout")
PROGRESS_INTERVAL_SECONDS = 10.0
PREDICTION_LABELS = {0: "good", 1: "bad"} 

LOGGER = logging.getLogger(SERVICE_NAME)


# ==================
# Structured logging 
# ==================


class JsonFormatter(logging.Formatter):

    def format(self, record: logging.LogRecord) -> str:
        entry: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=UTC)
            .isoformat(timespec="milliseconds")
            .replace("+00:00", "Z"),
            "level": record.levelname,
            "service": SERVICE_NAME,
            "event": getattr(record, "event", "log"),
            "message": record.getMessage(),
            **getattr(record, "fields", {}),
        }
        return json.dumps(entry, default=str)


def configure_logging(level: str) -> None:
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(JsonFormatter())
    LOGGER.handlers[:] = [handler]
    LOGGER.setLevel(level)
    LOGGER.propagate = False


def log_event(level: int, event: str, message: str, **fields: Any) -> None:
    LOGGER.log(level, message, extra={"event": event, "fields": fields})


# =============
# Input samples
# =============


@dataclass(frozen=True)
class SampleSet:
    """Feature rows (and optional reference labels) loaded from a preprocessed CSV."""

    columns: tuple[str, ...]
    rows: tuple[tuple[float, ...], ...]
    labels: tuple[int, ...] | None

    def random_batch(self, rng: random.Random, size: int) -> list[tuple[float, ...]]:
 
        return [self.rows[rng.randrange(len(self.rows))] for _ in range(size)]


def load_samples(path: Path, target_column: str = TARGET_COLUMN) -> SampleSet:

    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.reader(stream)
        header = next(reader)
        records = [row for row in reader if row]
    if not records:
        raise ValueError(f"No data rows found in {path}.")
    target_index = header.index(target_column) if target_column in header else None
    feature_indexes = [index for index in range(len(header)) if index != target_index]
    return SampleSet(
        columns=tuple(header[index] for index in feature_indexes),
        rows=tuple(tuple(float(record[index]) for index in feature_indexes) for record in records),
        labels=tuple(int(float(record[target_index])) for record in records) if target_index is not None else None,
    )


def build_payload(columns: Sequence[str], rows: Sequence[Sequence[Any]]) -> dict[str, Any]:

    return {"dataframe_split": {"columns": list(columns), "data": [list(row) for row in rows]}}


def encode_payload(columns: Sequence[str], rows: Sequence[Sequence[Any]]) -> bytes:
    return json.dumps(build_payload(columns, rows)).encode()



@dataclass(frozen=True)
class CallResult:

    request_id: str
    records: int
    status_code: int | None
    latency_seconds: float
    predictions: list[Any] | None = None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.status_code == 200 and self.predictions is not None

    @property
    def status_key(self) -> str:
        return str(self.status_code) if self.status_code is not None else (self.error or "error")


def send(
    http: httpx.Client,
    url: str,
    body: bytes,
    records: int,
    *,
    content_type: str = "application/json",
    fault_mode: str | None = None,
) -> CallResult:
    """POST ``body`` and classify the response; never raises on HTTP or transport errors."""
    request_id = uuid.uuid4().hex
    headers = {"Content-Type": content_type, REQUEST_ID_HEADER: request_id}
    if fault_mode:
        headers[FAULT_HEADER] = fault_mode
    started = time.perf_counter()
    try:
        response = http.post(url, content=body, headers=headers)
    except httpx.TimeoutException:
        result = CallResult(request_id, records, None, time.perf_counter() - started, error="client_timeout")
    except httpx.TransportError as exc:
        result = CallResult(request_id, records, None, time.perf_counter() - started, error=type(exc).__name__)
    else:
        result = _classify(response, request_id, records, time.perf_counter() - started)
    log_event(
        logging.DEBUG,
        "inference_call",
        "Inference call finished.",
        request_id=request_id,
        status=result.status_key,
        records=records,
        latency_ms=round(result.latency_seconds * 1000, 1),
    )
    return result


def _classify(response: httpx.Response, request_id: str, records: int, latency: float) -> CallResult:
    try:
        document = response.json()
    except ValueError:
        return CallResult(request_id, records, response.status_code, latency, error="non_json_response")
    if response.status_code == 200 and isinstance(document, dict):
        return CallResult(request_id, records, 200, latency, predictions=document.get("predictions"))
    error = document.get("error") if isinstance(document, dict) else None
    category = error.get("category") if isinstance(error, dict) else None
    return CallResult(request_id, records, response.status_code, latency, error=category or "http_error")


@dataclass
class RunStats:

    started: float = field(default_factory=time.monotonic)
    by_status: Counter[str] = field(default_factory=Counter)
    latencies: list[float] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def add(self, result: CallResult) -> None:
        with self._lock:
            self.by_status[result.status_key] += 1
            self.latencies.append(result.latency_seconds)

    def summary(self) -> str:
        with self._lock:
            total = sum(self.by_status.values())
            ordered = sorted(self.latencies)
            statuses = ", ".join(f"{key}: {value}" for key, value in sorted(self.by_status.items()))
        p95 = ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)] if ordered else float("nan")
        elapsed = time.monotonic() - self.started
        return f"{total} requests in {elapsed:.0f}s | statuses {{{statuses}}} | client-side p95 {p95 * 1000:.0f} ms"


class ProgressReporter:

    def __init__(self, label: str, stats: RunStats, interval: float = PROGRESS_INTERVAL_SECONDS) -> None:
        self._label, self._stats, self._interval = label, stats, interval
        self._next = time.monotonic() + interval

    def maybe_report(self) -> None:
        if time.monotonic() >= self._next:
            print(f"[{self._label}] {self._stats.summary()}", flush=True)
            self._next += self._interval


# =====
# Modes
# =====


def run_predict(http: httpx.Client, url: str, samples: SampleSet, rows: int, offset: int) -> int:

    batch = samples.rows[offset : offset + rows]
    if not batch:
        print(f"--offset {offset} is beyond the {len(samples.rows)} available rows.", file=sys.stderr)
        return 2
    result = send(http, url, encode_payload(samples.columns, batch), len(batch))
    print(f"POST {url}")
    print(f"{REQUEST_ID_HEADER} : {result.request_id}  (search for it in the gateway log)")
    print(f"HTTP status  : {result.status_code}  | client latency {result.latency_seconds * 1000:.1f} ms")
    if not result.ok:
        _explain_failure(url, result)
        return 1
    print(f"{'row':>5}  {'prediction':>10}  {'meaning':>8}  {'test-set label (reference only)':>32}")
    for index, prediction in enumerate(result.predictions or []):
        label = samples.labels[offset + index] if samples.labels else "-"
        print(f"{offset + index:>5}  {prediction:>10}  {PREDICTION_LABELS.get(prediction, '?'):>8}  {label:>32}")
    return 0


def _explain_failure(url: str, result: CallResult) -> None:
    if result.status_code is None:
        print(
            f"Cannot reach the gateway at {url} ({result.error}). Start it first, e.g. "
            "`docker compose up -d` or `python 3.prometheus_exporter.py`.",
            file=sys.stderr,
        )
    elif result.status_code == 503:
        print("The model server is not available yet; check `curl http://127.0.0.1:8000/readyz`.", file=sys.stderr)
    else:
        print(f"Request failed: {result.error}", file=sys.stderr)


def run_paced(
    http: httpx.Client,
    url: str,
    samples: SampleSet,
    *,
    label: str,
    duration: float,
    rate: float,
    batch_size: Callable[[random.Random], int],
    fault_for: Callable[[random.Random], str | None],
    seed: int,
) -> RunStats:
    """Sequential requests at ``rate`` per second for ``duration`` seconds (traffic and faults modes)."""
    rng, stats = random.Random(seed), RunStats()
    progress = ProgressReporter(label, stats)
    deadline, interval = time.monotonic() + duration, 1.0 / rate
    while time.monotonic() < deadline:
        tick = time.monotonic()
        rows = samples.random_batch(rng, batch_size(rng))
        stats.add(send(http, url, encode_payload(samples.columns, rows), len(rows), fault_mode=fault_for(rng)))
        progress.maybe_report()
        time.sleep(max(0.0, interval - (time.monotonic() - tick)))
    return stats


def run_load(
    url: str, samples: SampleSet, *, duration: float, concurrency: int, batch_size: int, timeout: float, seed: int
) -> RunStats:
    """``concurrency`` parallel workers sending large batches back-to-back (saturates the model server)."""
    stats, stop = RunStats(), threading.Event()
    deadline = time.monotonic() + duration

    def worker(worker_id: int) -> None:
        rng = random.Random(seed + worker_id)
        with httpx.Client(timeout=timeout) as http:
            while time.monotonic() < deadline and not stop.is_set():
                rows = samples.random_batch(rng, batch_size)
                stats.add(send(http, url, encode_payload(samples.columns, rows), len(rows)))

    pool = ThreadPoolExecutor(max_workers=concurrency)
    try:
        pending = {pool.submit(worker, worker_id) for worker_id in range(concurrency)}
        while pending:
            done, pending = wait(pending, timeout=PROGRESS_INTERVAL_SECONDS)
            for future in done:
                future.result()  # surface unexpected worker errors
            print(f"[load] {stats.summary()}", flush=True)
    finally:
        stop.set()  # lets workers finish their current request after Ctrl+C
        pool.shutdown(wait=True)
    return stats


def invalid_cases(samples: SampleSet) -> list[tuple[str, bytes, str]]:

    columns, row = list(samples.columns), list(samples.rows[0])
    return [
        ("missing column", encode_payload(columns[:-1], [row[:-1]]), "application/json"),
        ("non-numeric value", encode_payload(columns, [[*row[:-1], "not-a-number"]]), "application/json"),
        ("malformed JSON", b'{"dataframe_split": ', "application/json"),
        ("empty batch", encode_payload(columns, []), "application/json"),
        ("wrong content type", encode_payload(columns, [row]), "text/plain"),
    ]


def run_invalid(http: httpx.Client, url: str, samples: SampleSet) -> int:

    unexpected = 0
    for name, body, content_type in invalid_cases(samples):
        result = send(http, url, body, 0, content_type=content_type)
        rejected = result.status_code is not None and 400 <= result.status_code < 500
        unexpected += not rejected
        verdict = "OK (rejected)" if rejected else "UNEXPECTED"
        print(f"{name:<20} -> HTTP {result.status_code} ({result.error})  {verdict}")
    return 1 if unexpected else 0


# ======================
# Command-line interface
# ======================


def _positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be >= 1")
    return number


def _positive_float(value: str) -> float:
    number = float(value)
    if not number > 0:
        raise argparse.ArgumentTypeError("must be > 0")
    return number


def _ratio(value: str) -> float:
    number = float(value)
    if not 0 < number <= 1:
        raise argparse.ArgumentTypeError("must be in (0, 1]")
    return number


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="7.inference.py", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--url", default=DEFAULT_URL, help="gateway /invocations URL (default: %(default)s)")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA, help="preprocessed CSV with valid rows")
    parser.add_argument("--timeout", type=_positive_float, default=30.0, help="client timeout in seconds")
    parser.add_argument("--seed", type=int, default=42, help="random seed for row sampling")
    parser.add_argument("--log-level", default="WARNING", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    sub = parser.add_subparsers(dest="mode", metavar="{predict,traffic,load,invalid,faults}")

    predict = sub.add_parser("predict", help="one request, print predictions (default mode)")
    predict.add_argument("--rows", type=_positive_int, default=5)
    predict.add_argument("--offset", type=int, default=0)

    traffic = sub.add_parser("traffic", help="steady traffic for the dashboard")
    traffic.add_argument("--duration", type=_positive_float, default=600)
    traffic.add_argument("--rate", type=_positive_float, default=2.0, help="requests per second")
    traffic.add_argument("--min-batch", type=_positive_int, default=1)
    traffic.add_argument("--max-batch", type=_positive_int, default=50)

    load = sub.add_parser("load", help="saturating concurrent load (latency alert)")
    load.add_argument("--duration", type=_positive_float, default=240)
    load.add_argument("--concurrency", type=_positive_int, default=32)
    load.add_argument("--batch-size", type=_positive_int, default=1000)

    sub.add_parser("invalid", help="send invalid requests (validation path)")

    faults = sub.add_parser("faults", help="LOCAL TESTING ONLY: fault-injected traffic (failure-rate alert)")
    faults.add_argument("--mode", dest="fault_mode", choices=FAULT_MODES, default="schema-violation")
    faults.add_argument("--ratio", type=_ratio, default=0.5, help="share of requests carrying the fault header")
    faults.add_argument("--duration", type=_positive_float, default=240)
    faults.add_argument("--rate", type=_positive_float, default=2.0, help="requests per second")

    parser.set_defaults(mode="predict", rows=5, offset=0)
    return parser


def execute(args: argparse.Namespace, samples: SampleSet) -> int:

    if args.mode == "load":
        stats = run_load(
            args.url,
            samples,
            duration=args.duration,
            concurrency=args.concurrency,
            batch_size=args.batch_size,
            timeout=args.timeout,
            seed=args.seed,
        )
    else:
        with httpx.Client(timeout=args.timeout) as http:
            if args.mode == "predict":
                return run_predict(http, args.url, samples, args.rows, args.offset)
            if args.mode == "invalid":
                return run_invalid(http, args.url, samples)
            if args.mode == "traffic":
                stats = run_paced(
                    http,
                    args.url,
                    samples,
                    label="traffic",
                    duration=args.duration,
                    rate=args.rate,
                    seed=args.seed,
                    batch_size=lambda rng: rng.randint(args.min_batch, args.max_batch),
                    fault_for=lambda _: None,
                )
            else:
                stats = run_paced(
                    http,
                    args.url,
                    samples,
                    label="faults",
                    duration=args.duration,
                    rate=args.rate,
                    seed=args.seed,
                    batch_size=lambda rng: rng.randint(1, 10),
                    fault_for=lambda rng: args.fault_mode if rng.random() < args.ratio else None,
                )
    print(f"[{args.mode}] finished: {stats.summary()}")
    log_event(logging.INFO, "client_run_finished", stats.summary(), mode=args.mode)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.mode == "traffic" and args.min_batch > args.max_batch:
        print("--min-batch must not exceed --max-batch", file=sys.stderr)
        return 2
    configure_logging(args.log_level)
    try:
        samples = load_samples(args.data)
    except (OSError, ValueError, StopIteration) as exc:
        print(f"Cannot load samples from {args.data}: {exc or 'empty file'}", file=sys.stderr)
        return 2
    try:
        return execute(args, samples)
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
