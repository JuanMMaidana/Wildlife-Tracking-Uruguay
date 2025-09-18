import tempfile
import unittest
from pathlib import Path

from scripts.lib.species_map import SpeciesMapError, load_species_map

BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_DIR = BASE_DIR / "config"


class SpeciesMapTests(unittest.TestCase):
    def test_load_species_map_returns_all_species(self):
        classes_path = CONFIG_DIR / "classes.yaml"
        species = []
        with classes_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if stripped.startswith("- "):
                    species.append(stripped[2:])

        species_map = load_species_map(
            CONFIG_DIR / "species_map.yaml", valid_species=species
        )
        self.assertEqual(sorted(species_map.species), sorted(species))

    def test_duplicate_regex_raises_error(self):
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as tmp:
            tmp_path = Path(tmp.name)
            tmp.write(
                "patterns:\n"
                "  animal_one:\n"
                "    - '(?i)^example'\n"
                "  animal_two:\n"
                "    - '(?i)^example'\n"
            )
        try:
            with self.assertRaises(SpeciesMapError):
                load_species_map(tmp_path)
        finally:
            tmp_path.unlink(missing_ok=True)

    def test_unknown_species_raises_error(self):
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as tmp:
            tmp_path = Path(tmp.name)
            tmp.write(
                "patterns:\n"
                "  imaginary:\n"
                "    - '.*'\n"
            )
        try:
            with self.assertRaises(SpeciesMapError):
                load_species_map(tmp_path, valid_species=["margay"])
        finally:
            tmp_path.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
