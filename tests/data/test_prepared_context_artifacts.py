from __future__ import annotations

import json
from pathlib import Path
import unittest

from timelymt.data.prepared_context import load_prepared_context


ROOT = Path(__file__).parents[2]
ARTIFACT_ROOT = ROOT / "data" / "prepared_context"
SPLITS = json.loads((ROOT / "data" / "splits" / "experimental.json").read_text(encoding="utf-8"))["splits"]
EXPECTED = {split: tuple(SPLITS[split]) for split in ("train", "dev")}
INTENDED_SOURCES = {
    "ted-greg-brockman-chatgpt-potential": {
        "chatgpt-2022-launch-lead", "chatgpt-2023-plugins-lead",
    },
    "ted-joseph-redmon-computer-vision": {"yolo-2015-abstract", "yolo9000-2016-abstract"},
    "ted-stuart-russell-safe-ai": {"cirl-2016-abstract"},
    "ted-yejin-choi-ai-smart-stupid": {"delphi-2021-abstract"},
    "ted-sal-khan-ai-education": {"khanmigo-gpt4-2023-announcement-lead"},
    "ted-sims-witherspoon-ai-climate": {"deepmind-wind-energy-2019-lead"},
}


class PreparedContextArtifactTests(unittest.TestCase):
    def test_exact_train_and_dev_pools_are_local_valid_and_checksum_verified(self) -> None:
        discovered = {
            split: sorted(path.stem for path in (ARTIFACT_ROOT / split).glob("*.json"))
            for split in EXPECTED
        }
        self.assertEqual(discovered["train"], sorted(EXPECTED["train"]))
        self.assertEqual(discovered["dev"], sorted(EXPECTED["dev"]))
        self.assertFalse((ARTIFACT_ROOT / "test").exists())

        for split, talk_ids in EXPECTED.items():
            for talk_id in talk_ids:
                with self.subTest(split=split, talk_id=talk_id):
                    pool = load_prepared_context(ARTIFACT_ROOT / split / f"{talk_id}.json")
                    self.assertEqual(pool.talk_id, talk_id)
                    self.assertEqual(pool.split, split)
                    self.assertEqual(len(pool.sources), len({source.source_id for source in pool.sources}))
                    self.assertEqual({source.source_id for source in pool.eligible_sources()}, INTENDED_SOURCES.get(talk_id, set()))
                    self.assertNotIn(pool.metadata.title, {source.text for source in pool.sources})

    def test_manifest_matches_pool_identities_checksums_and_coverage(self) -> None:
        manifest = json.loads((ARTIFACT_ROOT / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["schema_version"], "prepared-context-v0")
        self.assertEqual(manifest["train_talks"], 12)
        self.assertEqual(manifest["dev_talks"], 3)
        self.assertEqual(manifest["pool_count"], 15)

        pools = [load_prepared_context(ARTIFACT_ROOT / item["path"]) for item in manifest["pools"]]
        self.assertEqual(len(pools), manifest["pool_count"])
        eligible = [source for pool in pools for source in pool.eligible_sources()]
        self.assertEqual(len(eligible), manifest["eligible_source_count"])
        self.assertEqual(sum(bool(pool.eligible_sources()) for pool in pools), manifest["talks_with_eligible_context"])
        self.assertEqual(sum(not pool.eligible_sources() for pool in pools), manifest["talks_without_eligible_context"])
        self.assertEqual(manifest["talks_with_eligible_context"] + manifest["talks_without_eligible_context"], 15)

        for item, pool in zip(manifest["pools"], pools, strict=True):
            self.assertEqual(item["talk_id"], pool.talk_id)
            self.assertEqual(item["split"], pool.split)
            self.assertEqual(
                item.get("sources", []),
                [{"source_id": source.source_id, "checksum": source.checksum} for source in pool.sources],
            )


if __name__ == "__main__":
    unittest.main()
