"""Create/validate the dedicated Policy V2 environment, then launch the runner."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any, NamedTuple, Sequence


ROOT = Path(__file__).resolve().parents[1]
ENV_NAME = "timelymt-v2"
PYTHON_VERSION = "3.10"
BOOTSTRAP_SCHEMA_VERSION = "1.0.0"
IDENTITY_PATH = ROOT / "outputs/experiments/policy-v2/local-env.json"
LOG_PATH = ROOT / "outputs/experiments/policy-v2/local-run.log"
CUDA_INDEX_URL = "https://download.pytorch.org/whl/cu126"
CUDA_TORCH = "torch==2.12.0"
REQUIREMENTS = ROOT / "requirements-policy-v2-local.txt"
EXPECTED_VERSIONS = {
    "torch_version": "2.12.0+cu126", "transformers_version": "4.57.6",
    "numpy_version": "2.2.6", "scikit_learn_version": "1.7.2",
    "sacrebleu_version": "2.6.0", "joblib_version": "1.5.3",
    "sentencepiece_version": "0.2.2",
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _log(message: str, *, dry_run: bool = False) -> None:
    if dry_run:
        return
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(f"{_utc_now()} {message}\n")


def _clean_environment(python: Path | None = None) -> dict[str, str]:
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    environment["PYTHONNOUSERSITE"] = "1"
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    if python is not None:
        prefix = python.parent
        runtime_paths = [
            prefix, prefix / "Library/mingw-w64/bin", prefix / "Library/usr/bin",
            prefix / "Library/bin", prefix / "Scripts", prefix / "bin",
        ]
        existing = environment.get("PATH", "").split(os.pathsep)
        seen = {str(path).casefold() for path in runtime_paths}
        environment["PATH"] = os.pathsep.join(
            [*(str(path) for path in runtime_paths), *(path for path in existing if path.casefold() not in seen)]
        )
    return environment


def resolve_conda_executable() -> Path:
    candidates = [
        os.environ.get("CONDA_EXE"), shutil.which("conda"),
        "C:/ProgramData/miniconda3/Scripts/conda.exe",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return Path(candidate).resolve()
    raise RuntimeError("Conda was not found. Install Miniconda or set CONDA_EXE, then rerun make policy-v2-local")


def conda_environments(conda: Path) -> list[Path]:
    process = subprocess.run(
        [str(conda), "env", "list", "--json"], check=True, capture_output=True, text=True,
        encoding="utf-8", errors="replace", env=_clean_environment(),
    )
    return [Path(value) for value in json.loads(process.stdout)["envs"]]


def prospective_environment_python(conda: Path) -> Path:
    process = subprocess.run(
        [str(conda), "info", "--json"], check=True, capture_output=True, text=True,
        encoding="utf-8", errors="replace", env=_clean_environment(),
    )
    envs_dirs = json.loads(process.stdout).get("envs_dirs", [])
    if not envs_dirs:
        raise RuntimeError("Conda did not report an environment directory")
    prefix = Path(envs_dirs[0]) / ENV_NAME
    return prefix / ("python.exe" if os.name == "nt" else "bin/python")


def resolve_environment_python(conda: Path) -> Path | None:
    for prefix in conda_environments(conda):
        if prefix.name.casefold() == ENV_NAME.casefold():
            python = prefix / ("python.exe" if os.name == "nt" else "bin/python")
            return python
    return None


def environment_create_command(conda: Path) -> list[str]:
    return [str(conda), "create", "--name", ENV_NAME, f"python={PYTHON_VERSION}", "pip", "--yes"]


def environment_install_commands(python: Path) -> list[list[str]]:
    return [
        [str(python), "-m", "pip", "install", "--index-url", CUDA_INDEX_URL, CUDA_TORCH],
        [str(python), "-m", "pip", "install", "--requirement", str(REQUIREMENTS)],
        [str(python), "-m", "pip", "install", "--no-deps", "--editable", str(ROOT)],
    ]


class Probe(NamedTuple):
    name: str
    timeout: int
    code: str


class ProbeError(RuntimeError):
    def __init__(self, message: str, *, timed_out: bool = False) -> None:
        super().__init__(message)
        self.timed_out = timed_out


PROBES = (
    Probe("Python", 30, "import json, sys; print(json.dumps({'environment_name': 'timelymt-v2', 'python_version': '.'.join(map(str, sys.version_info[:3])), 'python_executable': sys.executable}))"),
    Probe("torch import", 60, "import json, torch; print(json.dumps({'torch_version': torch.__version__, 'torch_cuda_version': torch.version.cuda}))"),
    Probe("CUDA", 90, "import json, torch; available = torch.cuda.is_available(); print(json.dumps({'cuda_available': available, 'gpu_name': torch.cuda.get_device_name(0) if available else None, 'gpu_capability': list(torch.cuda.get_device_capability(0)) if available else None}))"),
    Probe("transformers", 60, "import json, transformers; print(json.dumps({'transformers_version': transformers.__version__}))"),
    Probe("numpy", 30, "import json, numpy; print(json.dumps({'numpy_version': numpy.__version__}))"),
    Probe("sklearn", 60, "import json, sklearn; print(json.dumps({'scikit_learn_version': sklearn.__version__}))"),
    Probe("sacrebleu", 30, "import json, sacrebleu; print(json.dumps({'sacrebleu_version': sacrebleu.__version__}))"),
    Probe("joblib", 30, "import joblib, json; print(json.dumps({'joblib_version': joblib.__version__}))"),
    Probe("sentencepiece", 30, "import importlib.metadata as m, json, sentencepiece; print(json.dumps({'sentencepiece_version': m.version('sentencepiece')}))"),
    Probe("timelymt import", 30, "import json; from pathlib import Path; import timelymt; print(json.dumps({'timelymt_source': str(Path(timelymt.__file__).resolve())}))"),
)


def _probe_detail(stdout: str | bytes | None, stderr: str | bytes | None) -> str:
    def normalize(value: str | bytes | None) -> str:
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="replace").strip()
        return (value or "").strip()

    parts = [value for value in (normalize(stdout), normalize(stderr)) if value]
    return "\n".join(parts) or "no output"


def run_probe(python: Path, probe: Probe) -> dict[str, Any]:
    prefix = f"[ENV] {probe.name:.<20}"
    print(prefix, end=" ", flush=True)
    try:
        process = subprocess.run(
            [str(python), "-c", probe.code], capture_output=True, text=True,
            encoding="utf-8", errors="replace", env=_clean_environment(python), timeout=probe.timeout,
        )
    except subprocess.TimeoutExpired as error:
        print(f"TIMEOUT after {probe.timeout}s")
        detail = _probe_detail(error.stdout, error.stderr)
        raise ProbeError(
            f"environment probe '{probe.name}' timed out after {probe.timeout}s\n"
            f"Python executable: {python}\nCaptured output:\n{detail}",
            timed_out=True,
        ) from error
    if process.returncode:
        print("FAILED")
        raise ProbeError(
            f"environment probe '{probe.name}' failed with return code {process.returncode}\n"
            f"Python executable: {python}\nCaptured output:\n{_probe_detail(process.stdout, process.stderr)}"
        )
    try:
        value = json.loads(process.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError) as error:
        print("FAILED")
        raise ProbeError(
            f"environment probe '{probe.name}' returned invalid JSON\n"
            f"Python executable: {python}\nCaptured output:\n{_probe_detail(process.stdout, process.stderr)}"
        ) from error
    summary = next((str(item) for key, item in value.items() if key.endswith("_version")), "OK")
    if probe.name == "Python":
        summary = f"{value.get('python_version')} ({value.get('python_executable')})"
    if probe.name == "CUDA" and value.get("cuda_available"):
        summary = str(value.get("gpu_name"))
    print(f"OK {summary}" if summary != "OK" else "OK")
    return value


def validate_environment(python: Path) -> dict[str, Any]:
    if python is None or not python.is_file():
        raise RuntimeError("the Conda environment exists but has no Python executable")
    identity: dict[str, Any] = {}
    for probe in PROBES:
        identity.update(run_probe(python, probe))
    if not identity["python_version"].startswith(PYTHON_VERSION + "."):
        raise RuntimeError(f"{ENV_NAME} requires Python {PYTHON_VERSION}, found {identity['python_version']}")
    if not identity["cuda_available"] or not identity["torch_cuda_version"]:
        raise RuntimeError("CUDA-enabled PyTorch is required; CPU-only Torch and CPU EnViT5 fallback are forbidden")
    mismatches = {
        key: (expected, identity.get(key)) for key, expected in EXPECTED_VERSIONS.items()
        if identity.get(key) != expected
    }
    if mismatches:
        raise RuntimeError(f"pinned dependency version mismatch: {mismatches}")
    try:
        Path(identity["timelymt_source"]).resolve().relative_to(ROOT.resolve())
    except (KeyError, ValueError):
        raise RuntimeError(f"TimelyMT is not installed from this repository: {identity.get('timelymt_source')}") from None
    if identity["gpu_name"] != "NVIDIA GeForce RTX 3050 Ti Laptop GPU" or identity["gpu_capability"] != [8, 6]:
        raise RuntimeError(
            "unexpected CUDA GPU; expected NVIDIA GeForce RTX 3050 Ti Laptop GPU capability 8.6, "
            f"found {identity['gpu_name']} capability {identity['gpu_capability']}"
        )
    identity["bootstrap_schema_version"] = BOOTSTRAP_SCHEMA_VERSION
    identity["selected_python_version"] = PYTHON_VERSION
    return identity


def _write_identity(identity: dict[str, Any]) -> None:
    IDENTITY_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = IDENTITY_PATH.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(identity, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(IDENTITY_PATH)


def _run(command: Sequence[str]) -> None:
    subprocess.run(list(command), cwd=ROOT, check=True, env=_clean_environment())


def bootstrap(*, dry_run: bool = False) -> Path | None:
    conda = resolve_conda_executable()
    python = resolve_environment_python(conda)
    if dry_run:
        if python is None:
            print(f"[BOOTSTRAP] Environment missing -> WOULD CREATE {ENV_NAME}")
            return prospective_environment_python(conda)
        else:
            try:
                validate_environment(python)
            except RuntimeError as error:
                print(f"[BOOTSTRAP] {ENV_NAME} ........ WOULD REPAIR ({error})")
            else:
                print(f"[BOOTSTRAP] {ENV_NAME} ........ SKIP (valid)")
        return python
    created = python is None
    if created:
        print(f"[BOOTSTRAP] Creating Conda environment {ENV_NAME}")
        _log(f"ENV CREATE name={ENV_NAME} python={PYTHON_VERSION}")
        _run(environment_create_command(conda))
        python = resolve_environment_python(conda)
        if python is None:
            raise RuntimeError(f"Conda created {ENV_NAME} but its Python executable could not be resolved")
    try:
        identity = validate_environment(python)
    except RuntimeError as initial_error:
        if isinstance(initial_error, ProbeError) and initial_error.timed_out:
            raise RuntimeError(
                f"unable to validate {ENV_NAME}: {initial_error}. The environment was left intact."
            ) from initial_error
        print("[BOOTSTRAP] Installing CUDA PyTorch")
        print("[BOOTSTRAP] Installing pinned project dependencies")
        print("[BOOTSTRAP] Installing TimelyMT editable/source package")
        _log(f"ENV {'REPAIR' if not created else 'INSTALL'} name={ENV_NAME} reason={initial_error}")
        try:
            for command in environment_install_commands(python):
                _run(command)
            identity = validate_environment(python)
        except (OSError, subprocess.CalledProcessError, RuntimeError) as error:
            raise RuntimeError(
                f"unable to repair {ENV_NAME}: {error}. Do not delete it automatically; inspect it with "
                f"'{conda} run -n {ENV_NAME} python -c \"import torch; print(torch.__version__)\"'. "
                f"If the prefix is fundamentally corrupt, manually run '{conda} env remove -n {ENV_NAME}' and retry."
            ) from error
    else:
        print(f"[BOOTSTRAP] {ENV_NAME} ........ SKIP (valid)")
    _write_identity(identity)
    _log(f"ENV VALIDATE name={ENV_NAME} python={python} status=OK")
    _log(f"CUDA VALIDATE torch={identity['torch_version']} cuda={identity['torch_cuda_version']} gpu={identity['gpu_name']}")
    print("[BOOTSTRAP] OK")
    return python


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    try:
        python = bootstrap(dry_run=args.dry_run)
        assert python is not None
        command = [str(python), "-m", "timelymt.research.policy_v2_local"]
        if args.dry_run:
            if python.is_file():
                command.append("--dry-run")
            else:
                command = [sys.executable, "-m", "timelymt.research.policy_v2_local", "--dry-run", "--python-executable", str(python)]
        raise SystemExit(subprocess.run(command, cwd=ROOT, env=_clean_environment()).returncode)
    except (OSError, subprocess.CalledProcessError, RuntimeError, json.JSONDecodeError) as error:
        _log(f"bootstrap_error={type(error).__name__}: {error}", dry_run=args.dry_run)
        print(f"FAILED: {error}", file=sys.stderr)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
