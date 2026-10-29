from __future__ import annotations

import argparse
import logging
import platform
import re
import sys
from pathlib import Path

import yaml

LOGGER = logging.getLogger("export_pip_requirements")

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_OUTPUT = SCRIPT_DIR / "requirements-ci.txt"

CONDA_YAML_CANDIDATES: tuple[Path, ...] = (
    PROJECT_ROOT / "MLProject" / "conda.yaml",
    PROJECT_ROOT / "MLproject" / "conda.yaml",
    PROJECT_ROOT / "conda.yaml",
    SCRIPT_DIR / "conda.yaml",
    Path.cwd() / "MLProject" / "conda.yaml",
    Path.cwd() / "conda.yaml",
)

FALLBACK_DESCRIPTION = "Export the pinned pip requirements declared in a conda.yaml file."

PYTHON_PIN_PATTERN = re.compile(r"^python\s*={1,2}\s*(?P<version>[0-9][\w.*]*)\s*$")

EXACT_PIN_PATTERN = re.compile(r"^[A-Za-z0-9][\w.\-]*(\[[^\]]+\])?\s*==\s*\S")

class CondaSpecError(ValueError):

    pass

def resolve_conda_yaml(explicit: Path | None) -> Path:

    if explicit is not None:
        path = explicit.expanduser().resolve()
        if not path.is_file():
            raise CondaSpecError(f"No conda.yaml at {path}.")
        return path

    for candidate in CONDA_YAML_CANDIDATES:
        if candidate.is_file():
            return candidate.resolve()

    searched = "\n  - ".join(str(candidate) for candidate in CONDA_YAML_CANDIDATES)
    raise CondaSpecError(
        "Could not locate conda.yaml. Searched:\n  - "
        f"{searched}\n"
        "Pass the path explicitly, e.g. "
        "python ci/export_pip_requirements.py MLProject/conda.yaml"
    )


def load_dependencies(conda_yaml: Path) -> list:

    try:
        spec = yaml.safe_load(conda_yaml.read_text(encoding="utf-8"))
    except yaml.YAMLError as error:
        raise CondaSpecError(f"{conda_yaml} is not valid YAML: {error}") from error

    if not isinstance(spec, dict):
        raise CondaSpecError(f"{conda_yaml} must contain a YAML mapping at the top level.")
    dependencies = spec.get("dependencies")
    if not isinstance(dependencies, list):
        raise CondaSpecError(f"{conda_yaml} has no 'dependencies' list.")
    return dependencies


def python_pin(dependencies: list) -> str | None:

    for dep in dependencies:
        if not isinstance(dep, str):
            continue
        match = PYTHON_PIN_PATTERN.match(dep.strip())
        if match:
            return match.group("version")
    return None


def check_interpreter(expected: str | None, *, strict: bool) -> None:

    if expected is None:
        message = "conda.yaml does not pin python (expected 'python=X.Y.Z')."
        if strict:
            raise CondaSpecError(message)
        LOGGER.warning("%s Continuing without an interpreter check.", message)
        return

    actual = platform.python_version()
    expected_parts = [part for part in expected.split(".") if part != "*"]
    if actual.split(".")[: len(expected_parts)] == expected_parts:
        LOGGER.info("Interpreter %s matches conda.yaml python=%s.", actual, expected)
        return

    message = f"Runner Python is {actual} but conda.yaml pins python={expected}."
    if strict:
        raise CondaSpecError(message)
    LOGGER.warning("%s Exported requirements are unaffected.", message)

def pip_requirements(dependencies: list) -> list[str]:

    for dep in dependencies:
        if not (isinstance(dep, dict) and "pip" in dep):
            continue

        entries = dep["pip"]
        if not isinstance(entries, list):
            raise CondaSpecError("The pip section of conda.yaml must be a list.")

        requirements = [str(item).strip() for item in entries if str(item).strip()]
        if not requirements:
            raise CondaSpecError("The pip section of conda.yaml is empty.")

        unpinned = [
            req
            for req in requirements
            if not req.startswith("-") and not EXACT_PIN_PATTERN.match(req)
        ]
        if unpinned:
            raise CondaSpecError(
                "Unpinned pip requirements in conda.yaml (each needs '=='): "
                + ", ".join(unpinned)
            )
        return requirements

    raise CondaSpecError("conda.yaml has no pip section.")

def write_requirements(output: Path, requirements: list[str]) -> None:

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(requirements) + "\n", encoding="utf-8")

def cli_description() -> str:

    docstring = __doc__ or FALLBACK_DESCRIPTION
    return docstring.strip().splitlines()[0]

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=cli_description())
    parser.add_argument(
        "conda_yaml",
        type=Path,
        nargs="?",
        default=None,
        help="Path to the project's conda.yaml (default: auto-discovered).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Where to write the pip requirements file (default: %(default)s).",
    )
    parser.add_argument(
        "--strict-python",
        action="store_true",
        help="Fail when the running interpreter does not match the python pin.",
    )
    return parser.parse_args(argv)

def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s"
    )

def main(argv: list[str] | None = None) -> int:
    configure_logging()
    args = parse_args(argv)
    try:
        conda_yaml = resolve_conda_yaml(args.conda_yaml)
        LOGGER.info("Reading %s.", conda_yaml)
        dependencies = load_dependencies(conda_yaml)
        check_interpreter(python_pin(dependencies), strict=args.strict_python)
        requirements = pip_requirements(dependencies)
        write_requirements(args.output, requirements)
    except (OSError, CondaSpecError) as error:
        LOGGER.error("%s", error)
        return 1

    LOGGER.info(
        "Wrote %d pinned requirements to %s: %s",
        len(requirements),
        args.output,
        ", ".join(requirements),
    )
    return 0

if __name__ == "__main__":
    sys.exit(main())
