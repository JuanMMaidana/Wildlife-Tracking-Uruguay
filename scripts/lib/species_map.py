"""Utilities for loading and validating regex-based species mappings."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Pattern, Sequence

try:
    import yaml  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - fallback when PyYAML absent
    yaml = None


class SpeciesMapError(ValueError):
    """Raised when the species mapping configuration is invalid."""


@dataclass(frozen=True)
class CompiledPattern:
    species: str
    raw_pattern: str
    regex: Pattern[str]


class SpeciesMap:
    """Encapsulates compiled regex patterns for species resolution."""

    def __init__(self, order: Sequence[str], patterns: Dict[str, List[CompiledPattern]]):
        self._order = list(order)
        self._patterns = patterns

    @property
    def species(self) -> List[str]:
        """Return species labels in evaluation order."""
        return list(self._order)

    def match(self, filename: str) -> Optional[str]:
        """Return species for filename stem; first match wins.

        Args:
            filename: Raw filename or stem to evaluate.

        Returns:
            Species label when a rule matches, otherwise ``None``.

        Raises:
            SpeciesMapError: When multiple species match the same filename.
        """
        matches: List[str] = []
        for species in self._order:
            for compiled in self._patterns.get(species, []):
                if compiled.regex.search(filename):
                    matches.append(species)
                    break
        if not matches:
            return None
        first = matches[0]
        if any(match != first for match in matches[1:]):
            raise SpeciesMapError(
                f"Multiple species matched '{filename}': {matches}"
            )
        return first


def _dedupe_patterns(patterns: Iterable[str]) -> List[str]:
    seen = set()
    deduped: List[str] = []
    for pattern in patterns:
        if pattern in seen:
            continue
        seen.add(pattern)
        deduped.append(pattern)
    return deduped


def _parse_species_map(map_path: Path) -> Dict[str, List[str]]:
    """Parse species map YAML with optional fallback parser."""
    if yaml is not None:
        with map_path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        patterns_section = data.get("patterns", {})
        if not isinstance(patterns_section, dict):
            raise SpeciesMapError("'patterns' must be a mapping of species → regex list")
        return {
            species: list(patterns or [])
            for species, patterns in patterns_section.items()
        }

    # Minimal parser: assumes simple indentation-based YAML used in repo
    patterns: Dict[str, List[str]] = {}
    current_species: Optional[str] = None

    with map_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            stripped = raw_line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if stripped == "patterns:":
                continue
            if stripped.endswith(":") and not stripped.startswith("- "):
                current_species = stripped[:-1].strip()
                if not current_species:
                    raise SpeciesMapError("Empty species label in species_map.yaml")
                patterns[current_species] = []
                continue
            if stripped.startswith("- "):
                if current_species is None:
                    raise SpeciesMapError("Pattern defined before species label")
                pattern = stripped[2:].strip()
                if (pattern.startswith("'") and pattern.endswith("'")) or (
                    pattern.startswith('"') and pattern.endswith('"')
                ):
                    pattern = pattern[1:-1]
                patterns[current_species].append(pattern)
                continue
            raise SpeciesMapError(
                f"Unsupported line in species_map.yaml: {raw_line.rstrip()}"
            )

    return patterns


def load_species_map(
    path: Path | str,
    *,
    valid_species: Optional[Sequence[str]] = None,
) -> SpeciesMap:
    """Load and validate regex species mapping configuration."""
    map_path = Path(path)
    if not map_path.exists():
        raise SpeciesMapError(f"Species map not found: {map_path}")

    patterns_section = _parse_species_map(map_path)
    if not patterns_section:
        raise SpeciesMapError("species_map.yaml must define a 'patterns' section")

    order: List[str] = []
    compiled: Dict[str, List[CompiledPattern]] = {}
    seen_regex: Dict[str, str] = {}

    valid_set = set(valid_species) if valid_species else None

    for species, raw_patterns in patterns_section.items():
        if valid_set is not None and species not in valid_set:
            raise SpeciesMapError(f"Unknown species in map: {species}")
        if not isinstance(raw_patterns, list) or not raw_patterns:
            raise SpeciesMapError(f"Species '{species}' must map to a non-empty list")

        order.append(species)
        deduped = _dedupe_patterns(raw_patterns)
        compiled_patterns: List[CompiledPattern] = []

        for pattern in deduped:
            if not isinstance(pattern, str):
                raise SpeciesMapError(
                    f"Pattern for species '{species}' must be a string: {pattern!r}"
                )
            existing = seen_regex.get(pattern)
            if existing and existing != species:
                raise SpeciesMapError(
                    f"Regex pattern '{pattern}' defined for both '{existing}' and '{species}'"
                )
            try:
                regex = re.compile(pattern)
            except re.error as exc:
                raise SpeciesMapError(
                    f"Invalid regex '{pattern}' for species '{species}': {exc}"
                ) from exc
            seen_regex[pattern] = species
            compiled_patterns.append(CompiledPattern(species, pattern, regex))

        compiled[species] = compiled_patterns

    if valid_set:
        missing = sorted(valid_set.difference(order))
        if missing:
            raise SpeciesMapError(
                f"Patterns missing for species: {', '.join(missing)}"
            )

    return SpeciesMap(order, compiled)


__all__ = ["SpeciesMap", "SpeciesMapError", "load_species_map"]
