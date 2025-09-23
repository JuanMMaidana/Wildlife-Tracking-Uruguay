#!/usr/bin/env python3
"""Generate polished portfolio visuals from classification outputs."""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Dict, Iterable, List, Tuple, Union

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from collections import Counter, defaultdict
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
DOCS_PLOTS = ROOT / "docs" / "plots"

PREDICTIONS_PATH = ROOT / "experiments" / "exp_003_species" / "predictions_test.csv"
COUNTS_PATH = ROOT / "experiments" / "exp_004_counts" / "results.csv"
CROPS_ROOT = ROOT / "data"

DEFAULT_FONT_CANDIDATES = [
    "/System/Library/Fonts/Supplemental/Helvetica.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create confusion matrix, counts bar chart, and crop grid")
    parser.add_argument("--predictions", default=PREDICTIONS_PATH, help="Predictions CSV from eval script")
    parser.add_argument("--counts", default=COUNTS_PATH, help="Aggregated counts CSV")
    parser.add_argument("--manifest", default=CROPS_ROOT / "crops_manifest.csv", help="Manifest (optional, for labels)")
    parser.add_argument("--out-dir", default=DOCS_PLOTS, help="Output directory for plots")
    parser.add_argument("--grid-cols", type=int, default=5, help="Columns in crop grid")
    parser.add_argument("--grid-rows", type=int, default=3, help="Rows in crop grid")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for crop sampling")
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_font(size: int) -> ImageFont.FreeTypeFont:
    for path in DEFAULT_FONT_CANDIDATES:
        font_path = Path(path)
        if font_path.exists():
            return ImageFont.truetype(str(font_path), size)
    return ImageFont.load_default()


def read_predictions(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError("Predictions CSV is empty; run eval_classifier.py first.")
    return df


def read_counts(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError("Counts CSV is empty; run scripts/40_counts_by_species.py first.")
    return df


def prettify_label(label: str) -> str:
    return label.replace("_", "\n") if len(label) > 10 else label.replace("_", " ")


def plot_confusion_matrix(preds: pd.DataFrame, out_path: Path) -> None:
    species = sorted(set(preds["species_true"]).union(preds["species_pred"]))
    matrix = pd.crosstab(preds["species_true"], preds["species_pred"], dropna=False).reindex(index=species, columns=species, fill_value=0)

    fig, ax = plt.subplots(figsize=(12, 10))
    cax = ax.imshow(matrix.values, cmap="Greens")
    fig.colorbar(cax, fraction=0.046, pad=0.04)

    ax.set_title("Confusion Matrix (Test Split)", fontsize=18, pad=20)
    ax.set_xlabel("Predicted", fontsize=14)
    ax.set_ylabel("True", fontsize=14)

    ax.set_xticks(np.arange(len(species)))
    ax.set_yticks(np.arange(len(species)))
    ax.set_xticklabels([prettify_label(s) for s in species], fontsize=10)
    ax.set_yticklabels([prettify_label(s) for s in species], fontsize=10)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    max_val = matrix.values.max() or 1
    for i in range(len(species)):
        for j in range(len(species)):
            val = matrix.iloc[i, j]
            color = "white" if val > max_val * 0.6 else "black"
            ax.text(j, i, f"{val}", ha="center", va="center", color=color, fontsize=9)

    ensure_dir(out_path.parent)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_species_counts(counts_df: pd.DataFrame, out_path: Path) -> None:
    grouped = counts_df.groupby("species").agg(total_tracks=("n_tracks", "sum")).reset_index()
    grouped.sort_values("total_tracks", ascending=False, inplace=True)

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(range(len(grouped)), grouped["total_tracks"], color="#4C72B0")

    ax.set_title("Predicted Tracks per Species", fontsize=18, pad=15)
    ax.set_ylabel("# Tracks", fontsize=14)
    ax.set_xticks(range(len(grouped)))
    ax.set_xticklabels([prettify_label(s) for s in grouped["species"]], rotation=30, ha="right", fontsize=11)
    ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    for bar, value in zip(bars, grouped["total_tracks"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3, f"{value:.0f}", ha="center", va="bottom", fontsize=10)

    fig.tight_layout()
    ensure_dir(out_path.parent)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def _row_to_dict(row: Union[pd.Series, tuple]) -> Dict[str, str]:
    if isinstance(row, pd.Series):
        return row.to_dict()
    if hasattr(row, "_asdict"):
        return row._asdict()
    return dict(row)


def build_crop_grid(preds: pd.DataFrame, out_path: Path, rows: int, cols: int, seed: int, thumb_size: int = 224) -> None:
    rng = random.Random(seed)
    grouped: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for _, row in preds.iterrows():
        grouped[row["species_true"]].append(_row_to_dict(row))

    samples: List[Dict[str, str]] = []
    for species_rows in grouped.values():
        rng.shuffle(species_rows)
        samples.extend(species_rows[:3])

    needed = rows * cols
    if len(samples) < needed:
        remaining = preds.sample(n=min(needed - len(samples), len(preds)), random_state=seed)
        samples.extend([_row_to_dict(r) for r in remaining.itertuples(index=False)])

    samples = samples[:needed]
    width = cols * thumb_size
    height = rows * thumb_size
    grid = Image.new("RGB", (width, height), color="white")

    for idx, record in enumerate(samples):
        crop_rel = record.get("crop_path", "")
        # Handle both 'crops/...' and 'data/crops/...' formats
        if crop_rel.startswith("crops/"):
            crop_path = (ROOT / "data" / Path(crop_rel.replace("\\", "/"))).resolve()
        else:
            crop_path = (ROOT / Path(crop_rel.replace("\\", "/"))).resolve()
        if not crop_path.exists():
            continue
        img = Image.open(crop_path).convert("RGB")
        img = img.resize((thumb_size, thumb_size))
        x = (idx % cols) * thumb_size
        y = (idx // cols) * thumb_size
        grid.paste(img, (x, y))

    caption_font = load_font(26)
    canvas = Image.new("RGB", (width, height + 80), "white")
    canvas.paste(grid, (0, 80))
    draw = ImageDraw.Draw(canvas)
    draw.text((width / 2, 30), "Sample Crops", fill="black", font=caption_font, anchor="mm")

    ensure_dir(out_path.parent)
    canvas.save(out_path)


def main() -> int:
    args = parse_args()
    ensure_dir(Path(args.out_dir))

    preds = read_predictions(Path(args.predictions))
    counts = read_counts(Path(args.counts))

    plot_confusion_matrix(preds, Path(args.out_dir) / "confusion_matrix.png")
    plot_species_counts(counts, Path(args.out_dir) / "species_counts.png")
    build_crop_grid(preds, Path(args.out_dir) / "crop_grid.png", rows=args.grid_rows, cols=args.grid_cols, seed=args.seed)

    print(f"Generated plots under {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
