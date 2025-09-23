import csv
import tempfile
import unittest
from pathlib import Path

from experiments.exp_003_autolabel.summary import (
    SUMMARY_COLUMNS,
    compute_summary,
    read_manifest,
    write_report,
    write_summary,
)


class AutolabelSummaryTests(unittest.TestCase):
    def _write_manifest(self, rows):
        tmp = tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False)
        path = Path(tmp.name)
        fieldnames = [
            "video",
            "video_stem",
            "track_id",
            "species",
            "frame_index",
            "crop_path",
            "confidence",
            "source",
            "bbox_x",
            "bbox_y",
            "bbox_w",
            "bbox_h",
            "crop_x1",
            "crop_y1",
            "crop_x2",
            "crop_y2",
            "dwell_time_s",
        ]
        writer = csv.DictWriter(tmp, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
        tmp.close()
        return path

    def test_read_manifest_and_compute_summary(self):
        rows = [
            {
                "video": "margay001.mp4",
                "video_stem": "margay001",
                "track_id": "1",
                "species": "margay",
                "frame_index": "10",
                "crop_path": "data/crops/margay/margay001__tid1__f10.jpg",
                "confidence": "0.9",
                "source": "high",
                "bbox_x": "0",
                "bbox_y": "0",
                "bbox_w": "100",
                "bbox_h": "120",
                "crop_x1": "0",
                "crop_y1": "0",
                "crop_x2": "100",
                "crop_y2": "120",
                "dwell_time_s": "3.0",
            },
            {
                "video": "margay001.mp4",
                "video_stem": "margay001",
                "track_id": "1",
                "species": "margay",
                "frame_index": "12",
                "crop_path": "data/crops/margay/margay001__tid1__f12.jpg",
                "confidence": "0.88",
                "source": "high",
                "bbox_x": "1",
                "bbox_y": "1",
                "bbox_w": "100",
                "bbox_h": "120",
                "crop_x1": "1",
                "crop_y1": "1",
                "crop_x2": "101",
                "crop_y2": "121",
                "dwell_time_s": "3.0",
            },
            {
                "video": "capybara002.mp4",
                "video_stem": "capybara002",
                "track_id": "5",
                "species": "capybara",
                "frame_index": "20",
                "crop_path": "data/crops/capybara/capybara002__tid5__f20.jpg",
                "confidence": "0.7",
                "source": "high",
                "bbox_x": "10",
                "bbox_y": "10",
                "bbox_w": "80",
                "bbox_h": "90",
                "crop_x1": "10",
                "crop_y1": "10",
                "crop_x2": "90",
                "crop_y2": "100",
                "dwell_time_s": "6.5",
            },
        ]
        manifest_path = self._write_manifest(rows)
        try:
            loaded_rows = read_manifest(manifest_path)
            summary = compute_summary(loaded_rows)
        finally:
            manifest_path.unlink(missing_ok=True)

        summary_dict = {row["species"]: row for row in summary}
        self.assertEqual(summary_dict["margay"]["n_crops"], 2)
        self.assertEqual(summary_dict["margay"]["n_tracks"], 1)
        self.assertEqual(summary_dict["margay"]["n_videos"], 1)
        self.assertAlmostEqual(summary_dict["margay"]["median_track_len"], 3.0)
        self.assertEqual(summary_dict["capybara"]["n_crops"], 1)
        self.assertAlmostEqual(summary_dict["capybara"]["median_track_len"], 6.5)

    def test_write_outputs(self):
        summary_rows = [
            {
                "species": "test",
                "n_videos": 1,
                "n_tracks": 1,
                "n_crops": 2,
                "median_track_len": 5.0,
            }
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir)
            summary_path = write_summary(out_dir, summary_rows)
            report_path = write_report(out_dir, summary_rows)

            with summary_path.open() as handle:
                content = handle.read()
            self.assertIn("species", content)
            self.assertIn("test", content)

            with report_path.open() as handle:
                report = handle.read()
            self.assertIn("Auto-label Summary", report)
            self.assertIn("test", report)
            self.assertIn("Median Track Length", report)


if __name__ == "__main__":
    unittest.main()
