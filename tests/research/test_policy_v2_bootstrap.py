from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).parents[2]
SPEC = importlib.util.spec_from_file_location("policy_v2_bootstrap", ROOT / "scripts/policy_v2_bootstrap.py")
assert SPEC and SPEC.loader
bootstrap = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bootstrap)


def identity(*, cuda: bool = True):
    return {
        "environment_name": bootstrap.ENV_NAME, "python_version": "3.10.20",
        "python_executable": "C:/conda/envs/timelymt-v2/python.exe",
        "torch_version": "2.12.0+cu126", "torch_cuda_version": "12.6" if cuda else None,
        "cuda_available": cuda, "transformers_version": "4.57.6", "numpy_version": "2.2.6",
        "scikit_learn_version": "1.7.2", "sacrebleu_version": "2.6.0",
        "joblib_version": "1.5.3", "sentencepiece_version": "0.2.2",
        "gpu_name": "NVIDIA GeForce RTX 3050 Ti Laptop GPU" if cuda else None,
        "gpu_capability": [8, 6] if cuda else None,
        "timelymt_source": str(ROOT / "src/timelymt/__init__.py"),
    }


class BootstrapCommandTests(unittest.TestCase):
    def test_environment_creation_and_install_commands_are_dedicated(self):
        conda = Path("C:/conda/Scripts/conda.exe")
        create = bootstrap.environment_create_command(conda)
        self.assertEqual(create[create.index("--name") + 1], "timelymt-v2")
        self.assertIn("python=3.10", create)
        commands = bootstrap.environment_install_commands(Path("C:/conda/envs/timelymt-v2/python.exe"))
        self.assertIn("https://download.pytorch.org/whl/cu126", commands[0])
        self.assertIn("torch==2.12.0", commands[0])
        self.assertIn("--no-deps", commands[-1])
        self.assertFalse(any("smart-traffic" in token for command in commands for token in command))

    def test_explicit_python_resolution(self):
        with tempfile.TemporaryDirectory() as directory:
            prefix = Path(directory) / "timelymt-v2"
            prefix.mkdir()
            python = prefix / "python.exe"
            python.touch()
            with patch.object(bootstrap, "conda_environments", return_value=[prefix]):
                self.assertEqual(bootstrap.resolve_environment_python(Path("conda")), python)

    def test_prospective_python_uses_conda_environment_directory(self):
        output = json.dumps({"envs_dirs": ["C:/conda/envs", "D:/other-envs"]})
        result = type("Result", (), {"stdout": output})()
        with patch.object(bootstrap.subprocess, "run", return_value=result):
            self.assertEqual(
                bootstrap.prospective_environment_python(Path("conda")),
                Path("C:/conda/envs/timelymt-v2/python.exe"),
            )

    def test_cuda_and_gpu_identity_are_required(self):
        with patch.object(bootstrap, "run_probe", side_effect=lambda _python, _probe: identity(cuda=False)):
            with self.assertRaisesRegex(RuntimeError, "CUDA-enabled PyTorch"):
                bootstrap.validate_environment(Path(__file__))

    def test_validation_rejects_unpinned_dependency(self):
        value = identity()
        value["transformers_version"] = "5.0.0"
        with patch.object(bootstrap, "run_probe", side_effect=lambda _python, _probe: value):
            with self.assertRaisesRegex(RuntimeError, "pinned dependency version mismatch"):
                bootstrap.validate_environment(Path(__file__))

    def test_clean_environment_preserves_cache_and_conda_runtime_path(self):
        python = Path("C:/conda/envs/timelymt-v2/python.exe")
        with patch.dict(bootstrap.os.environ, {
            "HF_HUB_CACHE": "shared-cache", "PYTHONPATH": "foreign", "PATH": "C:/Windows/System32",
        }, clear=True):
            environment = bootstrap._clean_environment(python)
        self.assertEqual(environment["HF_HUB_CACHE"], "shared-cache")
        self.assertNotIn("PYTHONPATH", environment)
        self.assertEqual(environment["PYTHONDONTWRITEBYTECODE"], "1")
        paths = environment["PATH"].split(bootstrap.os.pathsep)
        self.assertEqual(paths[0], str(python.parent))
        self.assertIn(str(python.parent / "Library/bin"), paths)
        self.assertIn(str(python.parent / "Scripts"), paths)
        self.assertIn("C:/Windows/System32", paths)

    def test_staged_probe_order(self):
        self.assertEqual(
            [probe.name for probe in bootstrap.PROBES],
            ["Python", "torch import", "CUDA", "transformers", "numpy", "sklearn", "sacrebleu", "joblib", "sentencepiece", "timelymt import"],
        )

    def test_successful_probe_has_bounded_timeout(self):
        probe = bootstrap.Probe("numpy", 30, "import numpy")
        result = type("Result", (), {"returncode": 0, "stdout": '{"numpy_version": "2.2.6"}\n', "stderr": ""})()
        with patch.object(bootstrap.subprocess, "run", return_value=result) as run:
            self.assertEqual(bootstrap.run_probe(Path(__file__), probe), {"numpy_version": "2.2.6"})
        self.assertEqual(run.call_args.kwargs["timeout"], 30)
        self.assertEqual(run.call_args.args[0][1], "-c")
        self.assertNotIn("__main__", run.call_args.args[0])

    def test_probe_timeout_identifies_probe_and_propagates_output(self):
        probe = bootstrap.Probe("torch import", 60, "import torch")
        timeout = bootstrap.subprocess.TimeoutExpired("python", 60, output=b"partial stdout", stderr=b"driver stderr")
        with patch.object(bootstrap.subprocess, "run", side_effect=timeout):
            with self.assertRaises(bootstrap.ProbeError) as caught:
                bootstrap.run_probe(Path(__file__), probe)
        self.assertTrue(caught.exception.timed_out)
        self.assertIn("torch import", str(caught.exception))
        self.assertIn("60s", str(caught.exception))
        self.assertIn("partial stdout", str(caught.exception))
        self.assertIn("driver stderr", str(caught.exception))

    def test_failed_probe_propagates_stderr(self):
        probe = bootstrap.Probe("sklearn", 60, "import sklearn")
        result = type("Result", (), {"returncode": 1, "stdout": "", "stderr": "DLL load failed"})()
        with patch.object(bootstrap.subprocess, "run", return_value=result):
            with self.assertRaisesRegex(bootstrap.ProbeError, "DLL load failed"):
                bootstrap.run_probe(Path(__file__), probe)

    def test_cpu_only_torch_is_rejected(self):
        value = identity()
        value["torch_cuda_version"] = None
        with patch.object(bootstrap, "run_probe", side_effect=lambda _python, _probe: value):
            with self.assertRaisesRegex(RuntimeError, "CPU-only Torch"):
                bootstrap.validate_environment(Path(__file__))

    def test_valid_environment_is_skipped_and_identity_written(self):
        python = Path(__file__)
        with tempfile.TemporaryDirectory() as directory, patch.object(bootstrap, "resolve_conda_executable", return_value=Path("conda")), patch.object(
            bootstrap, "resolve_environment_python", return_value=python,
        ), patch.object(bootstrap, "validate_environment", return_value=identity()), patch.object(
            bootstrap, "IDENTITY_PATH", Path(directory) / "local-env.json",
        ), patch.object(bootstrap, "LOG_PATH", Path(directory) / "local-run.log"), patch.object(bootstrap, "_run") as run:
            self.assertEqual(bootstrap.bootstrap(), python)
            run.assert_not_called()
            document = json.loads((Path(directory) / "local-env.json").read_text(encoding="utf-8"))
            self.assertEqual(document["environment_name"], "timelymt-v2")
            self.assertEqual(document["gpu_capability"], [8, 6])

    def test_interrupted_environment_is_repaired_without_deletion(self):
        python = Path(__file__)
        with tempfile.TemporaryDirectory() as directory, patch.object(bootstrap, "resolve_conda_executable", return_value=Path("conda")), patch.object(
            bootstrap, "resolve_environment_python", return_value=python,
        ), patch.object(bootstrap, "validate_environment", side_effect=[RuntimeError("missing package"), identity()]), patch.object(
            bootstrap, "IDENTITY_PATH", Path(directory) / "local-env.json",
        ), patch.object(bootstrap, "LOG_PATH", Path(directory) / "local-run.log"), patch.object(bootstrap, "_run") as run:
            bootstrap.bootstrap()
            self.assertEqual(run.call_count, 3)
            self.assertFalse(any("remove" in command for call in run.call_args_list for command in call.args[0]))

    def test_validation_timeout_does_not_install_or_delete_environment(self):
        python = Path(__file__)
        timeout = bootstrap.ProbeError("torch import timed out", timed_out=True)
        with patch.object(bootstrap, "resolve_conda_executable", return_value=Path("conda")), patch.object(
            bootstrap, "resolve_environment_python", return_value=python,
        ), patch.object(bootstrap, "validate_environment", side_effect=timeout), patch.object(bootstrap, "_run") as run:
            with self.assertRaisesRegex(RuntimeError, "left intact"):
                bootstrap.bootstrap()
        run.assert_not_called()

    def test_all_validation_probes_have_finite_timeouts(self):
        self.assertTrue(all(0 < probe.timeout <= 90 for probe in bootstrap.PROBES))
        self.assertFalse(any("-m __main__" in probe.code for probe in bootstrap.PROBES))

    def test_dry_run_missing_environment_has_zero_mutation(self):
        with patch.object(bootstrap, "resolve_conda_executable", return_value=Path("conda")), patch.object(
            bootstrap, "resolve_environment_python", return_value=None,
        ), patch.object(bootstrap, "prospective_environment_python", return_value=Path("dedicated/python.exe")), patch.object(
            bootstrap, "_run",
        ) as run, patch.object(bootstrap, "_write_identity") as write:
            self.assertIsNotNone(bootstrap.bootstrap(dry_run=True))
            run.assert_not_called()
            write.assert_not_called()


if __name__ == "__main__":
    unittest.main()
