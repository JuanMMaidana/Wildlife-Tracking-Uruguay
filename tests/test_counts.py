import csv
import importlib.util
import tempfile
import unittest
from pathlib import Path

MODULE_PATH = Path("scripts/40_counts_by_species.py").resolve()


def load_counts_module():
    spec = importlib.util.spec_from_file_location("counts_by_species", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class CountsAggregationTests(unittest.TestCase):
    def test_aggregate_counts_majority_vote(self):
        module = load_counts_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            manifest_path = tmp / "manifest.csv"
            predictions_path = tmp / "preds.csv"

            with manifest_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "video",
                        "video_stem",
                        "track_id",
                        "species",
                        "frame_index",
                        "crop_path",
                        "dwell_time_s",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "video": "margay1.mp4",
                        "video_stem": "margay1",
                        "track_id": "1",
                        "species": "margay",
                        "frame_index": 0,
                        "crop_path": "crops/margay/sample1.jpg",
                        "dwell_time_s": 5.0,
                    }
                )
                writer.writerow(
                    {
                        "video": "margay1.mp4",
                        "video_stem": "margay1",
                        "track_id": "1",
                        "species": "margay",
                        "frame_index": 1,
                        "crop_path": "crops/margay/sample2.jpg",
                        "dwell_time_s": 5.0,
                    }
                )

            with predictions_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["crop_path", "species_pred", "confidence"])
                writer.writeheader()
                writer.writerow({"crop_path": "crops/margay/sample1.jpg", "species_pred": "margay", "confidence": 0.9})
                writer.writerow({"crop_path": "crops/margay/sample2.jpg", "species_pred": "margay", "confidence": 0.8})

            track_records, summary = module.aggregate_counts(
                module.load_csv(manifest_path),
                module.load_csv(predictions_path),
                min_confidence=0.0,
            )
            self.assertEqual(len(track_records), 1)
            self.assertEqual(track_records[0]["species_pred"], "margay")
            self.assertAlmostEqual(track_records[0]["confidence_mean"], 0.85, places=3)
            self.assertEqual(len(summary), 1)
            self.assertEqual(summary[0]["n_tracks"], 1)


if __name__ == "__main__":
    unittest.main()
