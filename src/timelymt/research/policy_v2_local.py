"""One-command, resumable Windows runner for Policy V2 through frozen DEV."""

from __future__ import annotations

import argparse
from contextlib import AbstractContextManager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any, Sequence

from .policy import VARIANTS
from .policy_v2 import (
    DATASET_CHECKSUM, ENCODER_MODEL_ID, ENCODER_REVISION, LOCAL_RUNTIME, SPLIT_CHECKSUM,
    THRESHOLDS, TRANSLATOR_FINGERPRINT, sha256_file, validate_v1_supervision,
)


ROOT = Path(__file__).parents[3]
ENV_NAME = "timelymt-v2"
ENV_PYTHON = Path(sys.executable).resolve()
V1_ARCHIVE = ROOT / "docs/archive/timelymt-checkpoint"
PSEUDO = ROOT / "data/policy/pseudo_labels"
EXPERIMENT = ROOT / "outputs/experiments/policy-v2"
LOCK_PATH = EXPERIMENT / ".local-run.lock"
LOG_PATH = EXPERIMENT / "local-run.log"
THRESHOLD_ARGS = tuple(f"{value:.2f}" for value in THRESHOLDS)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _directory_matches(source: Path, destination: Path) -> bool:
    source_files = {path.relative_to(source) for path in source.rglob("*") if path.is_file()}
    destination_files = {path.relative_to(destination) for path in destination.rglob("*") if path.is_file()}
    return source_files == destination_files and all(
        sha256_file(source / relative) == sha256_file(destination / relative) for relative in source_files
    )


def reconciliation_action(root: Path = ROOT) -> str:
    destination = root / "data/policy/pseudo_labels/train"
    source = root / "docs/archive/timelymt-checkpoint/data/policy/pseudo_labels/train"
    if not destination.exists():
        return "import"
    if _directory_matches(source, destination):
        return "skip"
    manifest_path = destination / "manifest.json"
    try:
        manifest = _load_json(manifest_path)
    except (OSError, UnicodeError, json.JSONDecodeError):
        return "backup"
    if manifest.get("artifact_status") == "full":
        raise RuntimeError("root TRAIN claims artifact_status=full but differs from immutable frozen V1")
    return "backup"


def reconcile_v1(root: Path = ROOT, *, dry_run: bool = False) -> str:
    action = reconciliation_action(root)
    destination = root / "data/policy/pseudo_labels/train"
    if dry_run:
        return action
    if action == "backup":
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        backup = root / "data/policy/pseudo_labels/_local_backup" / f"train.partial-{timestamp}"
        suffix = 1
        while backup.exists():
            backup = backup.with_name(f"train.partial-{timestamp}-{suffix}")
            suffix += 1
        backup.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(destination), str(backup))
        print(f"Backed up conflicting root TRAIN to: {backup}")
    return action


def verify_v1(root: Path = ROOT) -> None:
    expected = {"train": (12, 22018), "dev": (3, 5796)}
    for split_name, (talk_count, state_count) in expected.items():
        directory = root / "data/policy/pseudo_labels" / split_name
        manifest, rows = validate_v1_supervision(directory, split_name)
        if len(list(directory.glob("*.jsonl"))) != talk_count or len(rows) != state_count:
            raise RuntimeError(f"frozen V1 {split_name.upper()} count verification failed")
        if manifest["dataset_checksum"] != DATASET_CHECKSUM or manifest["split_checksum"] != SPLIT_CHECKSUM:
            raise RuntimeError(f"frozen V1 {split_name.upper()} checksum verification failed")
        if manifest["translator"]["config_fingerprint"] != TRANSLATOR_FINGERPRINT:
            raise RuntimeError(f"frozen V1 {split_name.upper()} translator verification failed")


def research_command(*arguments: str, python: Path = ENV_PYTHON) -> list[str]:
    return [str(python), "-m", "timelymt.research.cli", *arguments]


def rollout_command(variant: str, *, python: Path = ENV_PYTHON) -> list[str]:
    return [
        str(python), "-m", "timelymt.research.cli", "rollout-v2", "--split", "dev",
        "--variant", variant, "--thresholds", *THRESHOLD_ARGS, "--batch-size", "1",
        "--encoder-device", "cpu", "--encoder-dtype", "float32",
        "--policy-device", "cpu", "--policy-dtype", "float32",
        "--translator-device", "cuda", "--translator-dtype", "float16",
    ]


def train_command(variant: str, *, python: Path = ENV_PYTHON) -> list[str]:
    return research_command(
        "train-v2", "--variant", variant, "--encoder-device", "cpu", "--encoder-dtype", "float32",
        "--policy-device", "cpu", "--policy-dtype", "float32", python=python,
    )


def orchestration_commands(*, python: Path = ENV_PYTHON) -> list[list[str]]:
    commands = [research_command("import-v1", "--source", str(V1_ARCHIVE), python=python)]
    commands.extend(train_command(variant, python=python) for variant in VARIANTS)
    commands.extend(rollout_command(variant, python=python) for variant in VARIANTS)
    commands.extend([
        research_command("evaluate-v2", "--split", "dev", python=python), research_command("compare-v2", python=python),
        research_command("select-v2", "--split", "dev", python=python),
        research_command("freeze-v2", "--split", "dev", python=python),
    ])
    return commands


def _process_start_token(pid: int) -> int | None:
    if os.name != "nt":
        return None
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.GetProcessTimes.argtypes = (
        wintypes.HANDLE, ctypes.POINTER(wintypes.FILETIME), ctypes.POINTER(wintypes.FILETIME),
        ctypes.POINTER(wintypes.FILETIME), ctypes.POINTER(wintypes.FILETIME),
    )
    kernel32.GetProcessTimes.restype = wintypes.BOOL
    kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
    kernel32.CloseHandle.restype = wintypes.BOOL
    handle = kernel32.OpenProcess(0x1000, False, pid)
    if not handle:
        return None
    try:
        creation, exit_time, kernel, user = (wintypes.FILETIME() for _ in range(4))
        if not kernel32.GetProcessTimes(handle, creation, exit_time, kernel, user):
            return None
        return (creation.dwHighDateTime << 32) | creation.dwLowDateTime
    finally:
        kernel32.CloseHandle(handle)


def _lock_is_active(document: dict[str, Any]) -> bool:
    pid = document.get("pid")
    if not isinstance(pid, int):
        return False
    if os.name == "nt":
        token = _process_start_token(pid)
        return token is not None and token == document.get("process_start_token")
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError):
        return False


class LocalRunLock(AbstractContextManager["LocalRunLock"]):
    def __init__(self, path: Path = LOCK_PATH) -> None:
        self.path = path
        self.acquired = False

    def __enter__(self) -> "LocalRunLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        document = {
            "pid": os.getpid(), "process_start_token": _process_start_token(os.getpid()),
            "created_at": _utc_now(), "command": "make policy-v2-local",
        }
        while True:
            try:
                self.path.mkdir()
                (self.path / "owner.json").write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
                self.acquired = True
                return self
            except FileExistsError:
                owner = {}
                for _ in range(5):
                    try:
                        owner = _load_json(self.path / "owner.json")
                        break
                    except (OSError, UnicodeError, json.JSONDecodeError):
                        time.sleep(0.05)
                if _lock_is_active(owner):
                    raise RuntimeError(f"another policy-v2-local process is active (pid={owner['pid']})")
                if not owner and time.time() - self.path.stat().st_mtime < 5:
                    raise RuntimeError("another policy-v2-local process is initializing its lock")
                shutil.rmtree(self.path, ignore_errors=True)

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if self.acquired:
            shutil.rmtree(self.path, ignore_errors=True)


class Runner:
    def __init__(self, *, root: Path = ROOT, dry_run: bool = False) -> None:
        self.root, self.dry_run = root, dry_run
        self.log_path = root / "outputs/experiments/policy-v2/local-run.log"

    def log(self, message: str) -> None:
        line = f"{_utc_now()} {message}"
        if not self.dry_run:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.log_path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")

    def run_process(self, stage: str, command: Sequence[str], environment_name: str) -> None:
        self.log(f"stage_start={stage} environment={environment_name} command={subprocess.list2cmdline(command)}")
        environment = os.environ.copy()
        environment.pop("PYTHONPATH", None)
        environment["PYTHONNOUSERSITE"] = "1"
        process = subprocess.Popen(
            list(command), cwd=self.root, env=environment, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace",
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            with self.log_path.open("a", encoding="utf-8") as handle:
                handle.write(line)
        return_code = process.wait()
        self.log(f"stage_end={stage} return_code={return_code}")
        if return_code:
            raise RuntimeError(f"stage failed: {stage} (return code {return_code})")

    def stage(self, number: int, total: int, name: str, command: Sequence[str], environment_name: str) -> None:
        prefix = f"[{number}/{total}] {name:.<24}"
        if self.dry_run:
            print(f"{prefix} WOULD RUN [{environment_name}] {subprocess.list2cmdline(command)}")
            return
        print(f"{prefix} RUN")
        self.run_process(name, command, environment_name)
        print(f"{prefix} OK")


def train_will_run(variant: str) -> bool:
    from .policy_v2_runner import _valid_checkpoint

    return not _valid_checkpoint(variant)


def _minilm_cached() -> bool:
    try:
        from huggingface_hub.constants import HF_HUB_CACHE
    except ImportError:
        return False
    repository = "models--" + ENCODER_MODEL_ID.replace("/", "--")
    return (Path(HF_HUB_CACHE) / repository / "snapshots" / ENCODER_REVISION).is_dir()


def gpu_preflight() -> None:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in timelymt-v2; translator CPU fallback is forbidden")
    properties = torch.cuda.get_device_properties(0)
    free_bytes, total_bytes = torch.cuda.mem_get_info(0)
    gib = 1024 ** 3
    diagnostic = {
        "cuda_available": True, "torch_version": torch.__version__, "torch_cuda_version": torch.version.cuda,
        "device_name": properties.name, "capability": [properties.major, properties.minor],
        "total_vram": total_bytes, "free_vram": free_bytes,
        "allocated": torch.cuda.memory_allocated(0), "reserved": torch.cuda.memory_reserved(0),
    }
    print(json.dumps(diagnostic, sort_keys=True))
    cache_root = Path(os.environ.get("HF_HUB_CACHE", Path.home() / ".cache/huggingface/hub"))
    snapshot = cache_root / "models--VietAI--envit5-translation/snapshots/840bc88104d5a4277af740eaedb024df8c3093e7"
    weights = [path for pattern in ("*.safetensors", "pytorch_model*.bin") for path in snapshot.glob(pattern)]
    if weights:
        weight_bytes = sum(path.stat().st_size for path in weights)
        hard_minimum = weight_bytes + 256 * 1024 ** 2
        warning = weight_bytes * 2 + 512 * 1024 ** 2
        print(f"Pinned EnViT5 weight footprint={weight_bytes / gib:.2f} GiB")
        if free_bytes < hard_minimum:
            raise RuntimeError("free VRAM is below the pinned FP16 weights plus a minimal 256 MiB workspace")
        if free_bytes < warning:
            print("WARNING: VRAM is above the hard minimum but below the conservative runtime warning threshold")
    else:
        print("WARNING: unable to derive a weight-based VRAM threshold from the pinned local snapshot")


GPU_PREFLIGHT_CODE = (
    "from timelymt.research.policy_v2_local import gpu_preflight; gpu_preflight()"
)


def gpu_preflight_command(*, python: Path = ENV_PYTHON) -> list[str]:
    return [str(python), "-c", GPU_PREFLIGHT_CODE]


def _final_verify() -> None:
    verify_v1()
    from .policy_v2_runner import freeze_v2

    freeze_v2()
    frozen = _load_json(EXPERIMENT / "v2-frozen-config.json")
    if frozen.get("artifact_status") != "v2-dev-frozen-complete" or frozen.get("test_status") != "UNTOUCHED":
        raise RuntimeError("final V2 DEV freeze verification failed")


def run(*, dry_run: bool = False, python: Path = ENV_PYTHON) -> None:
    runner = Runner(dry_run=dry_run)
    action = reconcile_v1(dry_run=dry_run)
    runner.log(f"v1_reconciliation_action={action}")
    total = 12
    import_command = research_command("import-v1", "--source", str(V1_ARCHIVE), python=python)
    if dry_run:
        print(f"V1 reconciliation action: {action}")
        print(f"Dedicated environment: {ENV_NAME} ({python})")
        print(f"Frozen local runtime: {json.dumps(LOCAL_RUNTIME, sort_keys=True)}")
        runner.stage(1, total, "V1 import", import_command, ENV_NAME)
    else:
        if action == "skip" and (PSEUDO / "dev").is_dir():
            verify_v1()
            print(f"[1/{total}] {'V1 import':.<24} SKIP (valid frozen V1)")
        else:
            runner.stage(1, total, "V1 import", import_command, ENV_NAME)
            verify_v1()
    if not _minilm_cached():
        print("Downloading pinned MiniLM on first run" if not dry_run else "Pinned MiniLM absent: first real run will download it")
    number = 2
    for variant in VARIANTS:
        if not train_will_run(variant):
            disposition = "WOULD SKIP" if dry_run else "SKIP"
            print(f"[{number}/{total}] {f'Train V2 {variant}':.<24} {disposition} (valid checkpoint)")
            runner.log(f"stage_skip=Train V2 {variant} reason=valid_checkpoint")
            number += 1
            continue
        runner.stage(number, total, f"Train V2 {variant}", train_command(variant, python=python), f"{ENV_NAME} CPU/float32")
        number += 1
    rollout_commands = [rollout_command(variant, python=python) for variant in VARIANTS]
    if dry_run:
        print(f"GPU preflight: WOULD RUN in {ENV_NAME} before first rollout")
    else:
        runner.run_process("GPU preflight", gpu_preflight_command(python=python), f"{ENV_NAME} CUDA")
    for variant, command in zip(VARIANTS, rollout_commands, strict=True):
        runner.stage(number, total, f"Rollout DEV {variant}", command, f"{ENV_NAME} CPU/CPU/CUDA")
        number += 1
    for name, command in (
        ("Evaluate DEV", research_command("evaluate-v2", "--split", "dev", python=python)),
        ("Compare V1/V2", research_command("compare-v2", python=python)),
        ("Select DEV", research_command("select-v2", "--split", "dev", python=python)),
        ("Freeze DEV", research_command("freeze-v2", "--split", "dev", python=python)),
    ):
        runner.stage(number, total, name, command, ENV_NAME)
        number += 1
    if dry_run:
        print(f"[{number}/{total}] {'Final verification':.<24} WOULD RUN (DEV only, no mutation)")
    else:
        print(f"[{number}/{total}] {'Final verification':.<24} RUN")
        _final_verify()
        print(f"[{number}/{total}] {'Final verification':.<24} OK - STOP BEFORE TEST")


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--python-executable", type=Path, default=ENV_PYTHON, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.dry_run:
        run(dry_run=True, python=args.python_executable)
        return
    try:
        with LocalRunLock():
            run()
    except Exception as error:
        try:
            Runner().log(f"error={type(error).__name__}: {error}")
        except OSError:
            pass
        print(f"FAILED: {error}", file=sys.stderr)
        print("Fix the issue, then rerun:\nmake policy-v2-local", file=sys.stderr)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
