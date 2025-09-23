import csv
import json
import tempfile
import unittest
from pathlib import Path

from training.prepare_split import (
    assign_crop_splits,
    assign_video_splits,
    infer_video_species,
    write_crop_outputs,
    write_video_outputs,
)


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

    def test_assign_video_splits_respects_species_grouping(self):
        video_species = {f"v{i}": "margay" for i in range(4)}
        video_species.update({f"w{i}": "capybara" for i in range(4)})
        assignments = assign_video_splits(
            video_species,
            ratios={"train": 0.5, "validation": 0.25, "test": 0.25},
            seed=1,
        )
        self.assertEqual(len(assignments["train"]), 4)  # leftover goes to train
        self.assertEqual(len(assignments["validation"]), 2)
        self.assertEqual(len(assignments["test"]), 2)
        self.assertEqual(set(assignments["train"]) & set(assignments["validation"]), set())

    def test_assign_crop_splits_balances_species(self):
        rows = []
        for i in range(10):
            rows.append({"crop_path": f"crops/margay/m{i}.jpg", "species": "margay"})
        for i in range(10):
            rows.append({"crop_path": f"crops/bird/b{i}.jpg", "species": "bird"})
        splits = assign_crop_splits(rows, ratios={"train": 0.5, "validation": 0.25, "test": 0.25}, seed=2)
        total = sum(len(v) for v in splits.values())
        self.assertEqual(total, 20)
        self.assertGreaterEqual(len(splits["validation"]), 4)
        self.assertGreaterEqual(len(splits["test"]), 4)
        # Ensure both species present in each split
        for split_rows in splits.values():
            species = {row["species"] for row in split_rows}
            self.assertTrue({"margay", "bird"}.issubset(species))

    def test_write_outputs_creates_files(self):
        assignments = {"train": ["v1"], "validation": ["v2"], "test": []}
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir)
            json_path, csv_path = write_video_outputs(out_dir, assignments)
            self.assertTrue(json_path.exists())
            self.assertTrue(csv_path.exists())
            with json_path.open() as handle:
                data = json.load(handle)
            self.assertIn("train", data)
            with csv_path.open() as handle:
                reader = csv.reader(handle)
                rows = list(reader)
            self.assertGreaterEqual(len(rows), 2)  # header + entries

    def test_write_crop_outputs_creates_files(self):
        assignments = {
            "train": [{"crop_path": "crops/margay/m0.jpg", "species": "margay"}],
            "validation": [{"crop_path": "crops/bird/b0.jpg", "species": "bird"}],
            "test": [],
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir)
            json_path, csv_path = write_crop_outputs(out_dir, assignments)
            self.assertTrue(json_path.exists())
            self.assertTrue(csv_path.exists())
            with json_path.open() as handle:
                data = json.load(handle)
            self.assertEqual(data["train"], ["crops/margay/m0.jpg"])
            with csv_path.open() as handle:
                reader = csv.reader(handle)
                rows = list(reader)
            self.assertGreaterEqual(len(rows), 2)


if __name__ == "__main__":
    unittest.main()
