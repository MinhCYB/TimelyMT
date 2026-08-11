from __future__ import annotations

import ast
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).parents[2]
NOTEBOOK = ROOT / "notebooks/kaggle-policy-v2.ipynb"


class KagglePolicyV2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.document = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        cls.cells = cls.document["cells"]
        cls.sources = ["".join(cell["source"]) for cell in cls.cells]
        cls.text = "\n".join(cls.sources)

    def test_stage_order_and_stop(self):
        headings = [
            "## CONFIG", "## CLONE CURRENT V2 REPOSITORY", "## SOURCE-TREE IMPORT BOOTSTRAP",
            "## ENVIRONMENT + CUDA PRECHECK", "## KAGGLE AUTH",
            "## DOWNLOAD / VALIDATE IMMUTABLE V1 INPUT ARTIFACT", "## AUDIT V1 TRAIN/DEV SUPERVISION",
            "## DOWNLOAD / CACHE FROZEN MINILM ENCODER", "## PRECOMPUTE TRAIN EMBEDDINGS",
            "## TRAIN V2-P0", "## TRAIN V2-P1", "## TRAIN V2-P2", "## CHECKPOINT V2 MODELS",
            "## DEV V2 ROLLOUTS", "## CHECKPOINT DEV PREDICTIONS", "## V2 EVALUATION",
            "## V1/V2 COMPARISON", "## V2 DEV SELECTION", "## FREEZE V2 EXPLORATORY CONFIG",
            "## EXPORT + VERSION V2 DATASET", "# STOP BEFORE TEST",
        ]
        positions = [next(index for index, source in enumerate(self.sources) if heading in source) for heading in headings]
        self.assertEqual(positions, sorted(positions))
        self.assertEqual(positions[-1], len(self.cells) - 1)

    def test_pins_cli_and_expanded_layout(self):
        self.assertIn('ENCODER_REVISION = "e62509716f15c5fd03a6fd3156a4bc5e43f83f26"', self.text)
        self.assertIn('MODEL_REVISION = "840bc88104d5a4277af740eaedb024df8c3093e7"', self.text)
        self.assertIn('V1_CHECKPOINT_DATASET_REF = "iteams24/timelymt-research-checkpoints"', self.text)
        self.assertIn('V2_CHECKPOINT_DATASET_REF = "iteams24/timelymt-policy-v2-checkpoints"', self.text)
        self.assertIn("expanded_candidates", self.text)
        self.assertIn("timelymt-checkpoint", self.text)
        self.assertNotIn("python -m kaggle", self.text.lower())
        self.assertNotIn("datasets status", self.text.lower())

    def test_boundaries_export_and_test_safeguard(self):
        for boundary in ("v2-models-trained", "v2-dev-rollouts-complete", "v2-dev-frozen-complete"):
            self.assertIn(f'publish_v2_boundary("{boundary}")', self.text)
        self.assertIn("RUN_EMERGENCY_CHECKPOINT = False", self.text)
        self.assertIn("timelymt-policy-v2-artifacts.tar.gz", self.text)
        self.assertIn("embedding-cache", self.text)
        self.assertIn("pseudo_labels", self.text)
        self.assertNotIn('"--split", "test"', self.text.lower())

    def test_code_cells_parse(self):
        for index, cell in enumerate(self.cells):
            if cell["cell_type"] == "code":
                try:
                    ast.parse("".join(cell["source"]))
                except SyntaxError as error:
                    self.fail(f"notebook code cell {index} does not parse: {error}")


if __name__ == "__main__":
    unittest.main()
