from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd

LOGGER = logging.getLogger("smoke_test_serving")


class ServingCheckError(RuntimeError):

def wait_until_ready(base_url: str, timeout_s: float, interval_s: float = 3.0) -> None:

    deadline = time.monotonic() + timeout_s
    last_error = "no attempt made"
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(f"{base_url}/ping", timeout=5) as response:
                if response.status == 200:
                    LOGGER.info("Server ready at %s.", base_url)
                    return
                last_error = f"HTTP {response.status}"
        except (urllib.error.URLError, ConnectionError, TimeoutError) as error:
            last_error = str(error)
        time.sleep(interval_s)
    raise ServingCheckError(f"{base_url}/ping not ready after {timeout_s:.0f}s ({last_error}).")


def build_payload(data: Path, target: str, rows: int) -> tuple[bytes, int]:

    sample = pd.read_csv(data).drop(columns=[target]).head(rows)
    split = sample.to_dict(orient="split")
    body = {"dataframe_split": {"columns": split["columns"], "data": split["data"]}}
    return json.dumps(body).encode("utf-8"), len(sample)


def invoke(base_url: str, body: bytes) -> list:
   
    request = urllib.request.Request(f"{base_url}/invocations", data=body, method="POST",
                                     headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", "replace")[:500]
        raise ServingCheckError(f"/invocations returned HTTP {error.code}: {detail}") from error
    if not isinstance(payload, dict) or "predictions" not in payload:
        raise ServingCheckError(f"Unexpected response body: {str(payload)[:200]}")
    return payload["predictions"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--url", required=True, help="Base URL, e.g. http://127.0.0.1:5005")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--rows", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=180.0)
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
    args = parse_args()
    base_url = args.url.rstrip("/")
    try:
        wait_until_ready(base_url, args.timeout)
        body, expected = build_payload(args.data, args.target, args.rows)
        predictions = invoke(base_url, body)
        if len(predictions) != expected or not {int(p) for p in predictions} <= {0, 1}:
            raise ServingCheckError(f"Expected {expected} binary predictions, got {predictions}.")
    except (OSError, ValueError, ServingCheckError) as error:
        LOGGER.error("Serving smoke test failed: %s", error)
        return 1
    LOGGER.info("Serving OK: %d rows -> predictions %s", expected, predictions)
    return 0


if __name__ == "__main__":
    sys.exit(main())
