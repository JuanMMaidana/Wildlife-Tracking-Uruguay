import csv
import json
import tempfile
import unittest
from pathlib import Path

from training.prepare_split import assign_splits, infer_video_species, write_outputs


class PrepareSplitTests(unittest.TestCase):
    def test_infer_video_species_majority_vote(self):
        rows = [
            {"video_stem": "v1", "species": "margay"},
            {"video_stem": "v1", "species": "margay"},
            {"video_stem": "v2", "species": "capybara"},
            {"video_stem": "v2", "species": "capybara"},
            {"video_stem": "v2", "species": "capybara"},
        ]
        resolved = infer_video_species(rows)
        self.assertEqual(resolved["v1"], "margay")
        self.assertEqual(resolved["v2"], "capybara")

    def test_assign_splits_respects_species_grouping(self):
        video_species = {f"v{i}": "margay" for i in range(4)}
        video_species.update({f"w{i}": "capybara" for i in range(4)})
        assignments = assign_splits(video_species, ratios={"train": 0.5, "validation": 0.25, "test": 0.25}, seed=1)
        self.assertEqual(len(assignments["train"]), 4)  # leftover goes to train
        self.assertEqual(len(assignments["validation"]), 2)
        self.assertEqual(len(assignments["test"]), 2)
        self.assertEqual(set(assignments["train"]) & set(assignments["validation"]), set())

    def test_write_outputs_creates_files(self):
        assignments = {"train": ["v1"], "validation": ["v2"], "test": []}
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir)
            json_path, csv_path = write_outputs(out_dir, assignments)
            self.assertTrue(json_path.exists())
            self.assertTrue(csv_path.exists())
            with json_path.open() as handle:
                data = json.load(handle)
            self.assertIn("train", data)
            with csv_path.open() as handle:
                reader = csv.reader(handle)
                rows = list(reader)
            self.assertGreaterEqual(len(rows), 2)  # header + entries


if __name__ == "__main__":
    unittest.main()
