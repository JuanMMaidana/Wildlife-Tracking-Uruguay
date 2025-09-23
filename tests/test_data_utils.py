import csv
import tempfile
import unittest
from pathlib import Path

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None

if torch is not None:  # pragma: no branch
    from training import data_utils
else:  # pragma: no cover
    data_utils = None  # type: ignore


class DataUtilsTests(unittest.TestCase):
    def test_create_datasets_and_dataloaders_crop_split(self):
        if torch is None:
            self.skipTest("torch not available in current environment")
        if data_utils is None:
            self.skipTest("data_utils cannot be imported without torch")
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            data_root = tmp_path / "data"
            data_root.mkdir()
            crops_dir = data_root / "crops" / "margay"
            crops_dir.mkdir(parents=True)

            crop_path = crops_dir / "sample.jpg"
            # Create dummy image
            from PIL import Image

            Image.new("RGB", (64, 64), color=(255, 0, 0)).save(crop_path)

            manifest_path = data_root / "crops_manifest.csv"
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
                        "crop_path": "crops/margay/sample.jpg",
                    }
                )

            splits = {"train": ["crops/margay/sample.jpg"], "validation": [], "test": []}
            datasets, species_to_idx = data_utils.create_datasets(
                manifest_rows=data_utils.load_manifest_rows(manifest_path),
                splits=splits,
                split_type="crop",
                data_root=data_root,
                image_size=64,
            )
            self.assertEqual(len(datasets["train"]), 1)
            self.assertIn("margay", species_to_idx)

            loaders = data_utils.create_dataloaders(
                datasets,
                batch_size=1,
                num_workers=0,
                balance_classes=False,
            )
            images, labels, metadata = next(iter(loaders["train"]))
            self.assertEqual(images.shape[-1], 64)
            self.assertEqual(labels.shape[0], 1)
            self.assertEqual(metadata["video"][0], "margay1.mp4")
            self.assertTrue(isinstance(images, torch.Tensor))


if __name__ == "__main__":
    unittest.main()
