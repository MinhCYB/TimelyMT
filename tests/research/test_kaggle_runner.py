from __future__ import annotations

import json
from pathlib import Path
import unittest


ROOT = Path(__file__).parents[2]
NOTEBOOK = ROOT / "notebooks/kaggle-research-mvp.ipynb"


class KaggleRunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.document = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        cls.cells = cls.document["cells"]
        cls.sources = ["".join(cell["source"]) for cell in cls.cells]
        cls.text = "\n".join(cls.sources)

    def test_stage_cells_are_present_and_ordered(self):
        headings = [
            "## ENVIRONMENT SETUP", "## MODEL/CACHE SETUP", "## PRECHECK",
            "## TIMELYMT TRAIN PSEUDO", "## TIMELYMT DEV PSEUDO", "## MU TRAIN/DEV SUPERVISION",
            "## VALIDATE", "## TRAIN P0/P1/P2", "## TRAIN MU", "## DEV BASELINES",
            "## DEV LA-2", "## DEV MU ROLLOUT", "## DEV LEARNED ROLLOUT",
            "## DEV EVALUATE", "## DEV SELECT", "## FREEZE", "## EXPORT ARTIFACTS",
            "# STOP BEFORE TEST",
        ]
        positions = [next(index for index, source in enumerate(self.sources) if heading in source) for heading in headings]
        self.assertEqual(positions, sorted(positions))
        self.assertEqual(positions[-1], len(self.cells) - 1)

    def test_runner_is_pinned_gpu_safe_and_cli_driven(self):
        self.assertIn('MODEL_ID = "VietAI/envit5-translation"', self.text)
        self.assertIn('MODEL_REVISION = "840bc88104d5a4277af740eaedb024df8c3093e7"', self.text)
        self.assertIn('"transformers>=4.57.6,<5.0.0"', self.text)
        self.assertIn("torch.cuda.is_available()", self.text)
        self.assertIn('"timelymt.research.cli"', self.text)
        self.assertNotIn('cli("pseudo", "--split", "test"', self.text.lower())
        self.assertNotIn('cli("mu-supervision", "--split", "test"', self.text.lower())
        self.assertNotIn('cli("evaluate", "--split", "test"', self.text.lower())
        self.assertNotIn('cli("rollout", "--split", "test"', self.text.lower())

    def test_export_contains_required_artifacts_not_translator_weights(self):
        for path in (
            "data/policy/pseudo_labels", "data/policy/mu_zhang2020", "checkpoints/policy",
            "outputs/experiments/research-mvp",
        ):
            self.assertIn(path, self.text)
        export_cell = next(source for source in self.sources if "ARTIFACT_DIRS" in source)
        self.assertNotIn("huggingface", export_cell.lower())
        self.assertNotIn("outputs/translator", export_cell.lower())


if __name__ == "__main__":
    unittest.main()
