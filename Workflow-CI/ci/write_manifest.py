from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import sys
from datetime import UTC, datetime
from pathlib import Path

LOGGER = logging.getLogger("write_manifest")
SCHEMA_VERSION = 1
REQUIRED_GITHUB_ENV = ("GITHUB_SERVER_URL", "GITHUB_REPOSITORY", "GITHUB_SHA",
                       "GITHUB_REF_NAME", "GITHUB_EVENT_NAME", "GITHUB_WORKFLOW",
                       "GITHUB_RUN_ID", "GITHUB_RUN_ATTEMPT", "GITHUB_RUN_NUMBER")
DIGEST_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")


class ManifestError(RuntimeError):

def github_context() -> dict[str, str]:

    missing = [name for name in REQUIRED_GITHUB_ENV if not os.environ.get(name)]
    if missing:
        raise ManifestError(f"Missing GitHub Actions variables: {', '.join(missing)}")
    return {name: os.environ[name] for name in REQUIRED_GITHUB_ENV}


def sha256_of(path: Path) -> str:

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    if not path.is_file():
        raise ManifestError(f"Required input {path} does not exist.")
    return json.loads(path.read_text(encoding="utf-8"))


def build_manifest(args: argparse.Namespace) -> dict:

    ctx = github_context()
    run = read_json(args.run_facts)
    if not args.model_archive.is_file():
        raise ManifestError(f"Model archive {args.model_archive} does not exist.")
    repo_url = f"{ctx['GITHUB_SERVER_URL']}/{ctx['GITHUB_REPOSITORY']}"
    archive_name = args.model_archive.name
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "source": {
            "repository": ctx["GITHUB_REPOSITORY"],
            "commit_sha": ctx["GITHUB_SHA"],
            "ref": ctx["GITHUB_REF_NAME"],
            "event": ctx["GITHUB_EVENT_NAME"],
        },
        "workflow": {
            "name": ctx["GITHUB_WORKFLOW"],
            "run_id": ctx["GITHUB_RUN_ID"],
            "run_attempt": ctx["GITHUB_RUN_ATTEMPT"],
            "run_number": ctx["GITHUB_RUN_NUMBER"],
            "run_url": f"{repo_url}/actions/runs/{ctx['GITHUB_RUN_ID']}",
        },
        "mlflow": {key: run[key] for key in (
            "run_id", "run_name", "experiment_id", "status", "git_commit",
            "model_artifact_path", "model_uri", "model_location")},
        "metrics": run["metrics"],
        "params": run["params"],
        "verification": read_json(args.verification),
        "model_archive": {
            "file": archive_name,
            "sha256": sha256_of(args.model_archive),
            "size_bytes": args.model_archive.stat().st_size,
            "download_url": f"{repo_url}/releases/download/{args.release_tag}/{archive_name}",
        },
        "release": {
            "tag": args.release_tag,
            "url": f"{repo_url}/releases/tag/{args.release_tag}",
        },
        "docker": {
            "repository": args.image_repository,
            "tags": args.image_tag,
            "pushed": False,
            "digest": None,
        },
    }


def record_image(manifest_path: Path, digest: str) -> dict:

    if not DIGEST_PATTERN.match(digest):
        raise ManifestError(f"'{digest}' is not a sha256 image digest.")
    manifest = read_json(manifest_path)
    docker = manifest["docker"]
    docker.update({
        "pushed": True,
        "digest": digest,
        "pinned_reference": f"{docker['repository']}@{digest}",
    })
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    create = sub.add_parser("create", help="Write a new manifest.")
    create.add_argument("--run-facts", type=Path, required=True)
    create.add_argument("--verification", type=Path, required=True)
    create.add_argument("--model-archive", type=Path, required=True)
    create.add_argument("--release-tag", required=True)
    create.add_argument("--image-repository", required=True)
    create.add_argument("--image-tag", action="append", required=True)
    create.add_argument("--output", type=Path, required=True)

    image = sub.add_parser("record-image", help="Record the pushed image digest.")
    image.add_argument("--manifest", type=Path, required=True)
    image.add_argument("--digest", required=True)
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
    args = parse_args()
    try:
        if args.command == "create":
            manifest, target = build_manifest(args), args.output
        else:
            manifest, target = record_image(args.manifest, args.digest), args.manifest
    except (OSError, KeyError, json.JSONDecodeError, ManifestError) as error:
        LOGGER.error("%s", error)
        return 1
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    LOGGER.info("%s: %s", args.command, target)
    return 0


if __name__ == "__main__":
    sys.exit(main())
