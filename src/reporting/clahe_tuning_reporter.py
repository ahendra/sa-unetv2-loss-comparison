"""CLAHE Parameter Tuning Study.

Sequential grid search:
  Step A — Fix tileGridSize=8, vary clipLimit ∈ [1.0, 1.5, 2.0, 3.0, 4.0]
            Pick best clipLimit via weighted composite score (F1+AUC+Sensitivity+β0).
  Step B — Fix best_clip from Step A, vary tileGridSize ∈ [4, 16]
            (tileGridSize=8 is already covered in Step A, so +2 runs only)

Total: 5 + 2 = 7 full pipeline runs per dataset × 2 datasets = 14 training runs.

Each run: augment → train (BCE+MCC proxy) → evaluate → save results.
Skip any run where both weights and result JSON already exist.

Output (per invocation):
  results/<dataset>/clahe_tune_<tag>_results.json  — per-combination metrics
  <out_dir>/clahe_tuning_results.json              — combined summary + recommendation
  <out_dir>/clahe_tuning_drive.png                 — bar chart per metric (DRIVE)
  <out_dir>/clahe_tuning_stare.png                 — bar chart per metric (STARE)
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from config import DriveConfig, StareConfig, RESULTS_DIR
from src.data import RetinalAugmentationRunner, DriveDataLoader, StareDataLoader
from src.evaluation import ModelEvaluator
from src.losses import get_loss_function
from src.preprocessing import build_pipeline
from src.training import ModelTrainer
from src.training.trainer import set_global_seed


# ── Search space ──────────────────────────────────────────────────────────────

STEP_A_CLIPS: List[float] = [1.0, 1.5, 2.0, 3.0, 4.0]
STEP_A_TILE:  int         = 8
STEP_B_TILES: List[int]   = [4, 16]   # 8 already done in Step A

PROXY_LOSS = "bce_mcc"
DATASETS   = ["drive", "stare"]

_METRIC_KEYS = [
    "accuracy", "sensitivity", "specificity", "auc",
    "mcc", "f1", "jaccard", "cldice", "betti0_error", "betti1_error",
]
_LOWER_IS_BETTER = {"betti0_error", "betti1_error"}

# Weights for composite scoring used to select the best combination.
# Each metric is min-max normalised per-dataset before weighting so that
# different absolute value ranges (e.g. % vs count) do not skew the result.
SCORING_WEIGHTS: Dict[str, float] = {
    "f1":           0.35,   # primary segmentation metric
    "auc":          0.25,   # overall discriminative ability
    "sensitivity":  0.20,   # vessel recall (clinically important)
    "betti0_error": 0.20,   # topological integrity (lower is better)
}


# ── Naming helpers ────────────────────────────────────────────────────────────

def _tag(clip: float, tile: int) -> str:
    """Human-readable tag, e.g. 'clip1.5_tile8'."""
    return f"clip{clip:.1f}_tile{tile}"


def _dir_tag(clip: float, tile: int) -> str:
    """Filesystem-safe tag (no dots), e.g. 'clip1p5_tile8'."""
    return f"clip{clip:.1f}_tile{tile}".replace(".", "p")


def _loss_key(clip: float, tile: int) -> str:
    """Key used for weight and result file naming."""
    return f"clahe_tune_{_dir_tag(clip, tile)}"


# ── Config factory ────────────────────────────────────────────────────────────

def _make_cfg(
    dataset: str, clip: float, tile: int, n_epochs: int
) -> Union[DriveConfig, StareConfig]:
    """Instantiate a config with custom CLAHE params and tuning-specific aug dirs."""
    if dataset == "drive":
        cfg = DriveConfig(
            preprocessing_mode="clahe",
            clahe_clip_limit=clip,
            clahe_tile_grid=tile,
        )
    else:
        cfg = StareConfig(
            preprocessing_mode="clahe",
            clahe_clip_limit=clip,
            clahe_tile_grid=tile,
        )

    # cfg.aug_dir is now ".../DRIVE/aug_clahe" — parent is the dataset root dir
    dataset_root = Path(cfg.aug_dir).parent
    aug_base = dataset_root / f"aug_{_dir_tag(clip, tile)}"
    cfg.aug_dir          = str(aug_base)
    cfg.aug_train_images = str(aug_base / "train"    / "images")
    cfg.aug_train_labels = str(aug_base / "train"    / "labels")
    cfg.aug_val_images   = str(aug_base / "validate" / "images")
    cfg.aug_val_labels   = str(aug_base / "validate" / "labels")

    cfg.epochs = n_epochs
    return cfg


# ── Reporter ──────────────────────────────────────────────────────────────────

class ClaheTuningReporter:
    """Sequential CLAHE parameter search: augment + train + evaluate per combination."""

    def __init__(self, out_dir: Path):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

    # ── Entry point ───────────────────────────────────────────────────────────

    def run(self, n_epochs: int = 150) -> None:
        all_results: Dict[str, Dict[str, Dict]] = {d: {} for d in DATASETS}

        # ── Step A: vary clipLimit, fixed tile=8 ─────────────────────────────
        print("\n" + "=" * 62)
        print("  CLAHE TUNING — STEP A")
        print(f"  tileGridSize={STEP_A_TILE} (fixed) | clipLimit: {STEP_A_CLIPS}")
        print("=" * 62)

        for clip in STEP_A_CLIPS:
            tag = _tag(clip, STEP_A_TILE)
            print(f"\n  ── clipLimit={clip}, tileGridSize={STEP_A_TILE} ──")
            for dataset in DATASETS:
                metrics = self._run_one(dataset, clip, STEP_A_TILE, n_epochs)
                all_results[dataset][tag] = metrics
                self._print_brief(dataset, metrics)

        step_a_best_clip = self._pick_best_clip(all_results, STEP_A_CLIPS, STEP_A_TILE)
        print(f"\n  ✓ Step A best clipLimit: {step_a_best_clip}  "
              f"(composite score across DRIVE+STARE)")

        # ── Step B: vary tileGridSize, fixed best_clip ────────────────────────
        print("\n" + "=" * 62)
        print("  CLAHE TUNING — STEP B")
        print(f"  clipLimit={step_a_best_clip} (best from A) | "
              f"tileGridSize: {[STEP_A_TILE] + STEP_B_TILES}  "
              f"(tile={STEP_A_TILE} already done)")
        print("=" * 62)

        for tile in STEP_B_TILES:
            tag = _tag(step_a_best_clip, tile)
            print(f"\n  ── clipLimit={step_a_best_clip}, tileGridSize={tile} ──")
            for dataset in DATASETS:
                metrics = self._run_one(dataset, step_a_best_clip, tile, n_epochs)
                all_results[dataset][tag] = metrics
                self._print_brief(dataset, metrics)

        # ── Pick final winner ─────────────────────────────────────────────────
        best_clip, best_tile = self._pick_best_overall(
            all_results, step_a_best_clip
        )

        # ── Save + plot + summary ─────────────────────────────────────────────
        self._save_combined(all_results, best_clip, best_tile, step_a_best_clip)
        for dataset in DATASETS:
            self._plot(dataset, all_results[dataset], best_clip, best_tile)
        self._print_summary(all_results, best_clip, best_tile, step_a_best_clip)

    # ── Per-combination pipeline ──────────────────────────────────────────────

    def _run_one(
        self, dataset: str, clip: float, tile: int, n_epochs: int
    ) -> Dict:
        lkey = _loss_key(clip, tile)

        # Skip if result already saved from a previous run
        cached = self._load_cached(dataset, clip, tile)
        if cached is not None:
            print(f"    [{dataset.upper()}] [SKIP] Hasil sudah ada — dimuat dari cache.")
            return cached

        cfg = _make_cfg(dataset, clip, tile, n_epochs)

        self._augment(cfg, dataset)
        self._train(cfg, lkey)
        metrics = self._evaluate(cfg, lkey)

        # Cache individual result
        self._cache_result(dataset, clip, tile, metrics)
        return metrics

    # ── Augmentation ─────────────────────────────────────────────────────────

    def _augment(self, cfg: Union[DriveConfig, StareConfig], dataset: str) -> None:
        aug_train = Path(cfg.aug_train_images)
        if aug_train.exists() and any(aug_train.iterdir()):
            print(f"    [SKIP-AUG] Augmentasi sudah ada: {Path(cfg.aug_dir).name}")
            return

        print(f"    Augmentasi → {Path(cfg.aug_dir).name} ...")
        pipeline = build_pipeline("clahe", cfg.clahe_clip_limit, cfg.clahe_tile_grid)

        if dataset == "drive":
            def label_fn(img_name: str) -> str:
                stem = img_name.split('_')[0]
                for ext in ('.gif', '.png'):
                    cand = os.path.join(cfg.train_labels, f"{stem}_manual1{ext}")
                    if os.path.exists(cand):
                        return f"{stem}_manual1{ext}"
                return f"{stem}_manual1.gif"
        else:
            def label_fn(img_name: str) -> str:
                base = os.path.splitext(img_name)[0]
                return f"{base}.ah.ppm"

        RetinalAugmentationRunner().run(
            src_img_dir            = cfg.train_images,
            src_lbl_dir            = cfg.train_labels,
            aug_base_dir           = cfg.aug_dir,
            label_suffix_fn        = label_fn,
            preprocessing_pipeline = pipeline,
        )

    # ── Training ─────────────────────────────────────────────────────────────

    def _train(self, cfg: Union[DriveConfig, StareConfig], lkey: str) -> None:
        trainer = ModelTrainer(cfg)
        if trainer.weights_exist(lkey):
            print(f"    [SKIP-TRAIN] Bobot sudah ada: {lkey}")
            return

        print(f"    Training {lkey} ...")
        if isinstance(cfg, DriveConfig):
            loader  = DriveDataLoader(cfg)
            x_train, y_train = loader.load_train()
            x_val,   y_val   = loader.load_validate()
        else:
            loader  = StareDataLoader(cfg)
            x_train, y_train = loader.load_train()
            x_val,   y_val   = loader.load_validate()

        set_global_seed()
        loss_fn = get_loss_function(PROXY_LOSS)
        trainer.train(lkey, loss_fn, x_train, y_train, x_val, y_val)

    # ── Evaluation ────────────────────────────────────────────────────────────

    def _evaluate(
        self, cfg: Union[DriveConfig, StareConfig], lkey: str
    ) -> Dict:
        trainer   = ModelTrainer(cfg)
        evaluator = ModelEvaluator(cfg)

        if evaluator.results_exist(lkey):
            print(f"    [SKIP-EVAL] Hasil eval sudah ada: {lkey}")
            return evaluator.load_results(lkey) or {}

        if not trainer.weights_exist(lkey):
            print(f"    [WARN] Bobot tidak ditemukan untuk: {lkey}")
            return {}

        print(f"    Evaluasi {lkey} ...")
        model = trainer.load_weights(lkey)

        if isinstance(cfg, DriveConfig):
            loader = DriveDataLoader(cfg)
            x_test, y_test, masks = loader.load_test()
            restore_fn = loader.restore_predictions
        else:
            loader = StareDataLoader(cfg)
            x_test, y_test = loader.load_test()
            masks      = None
            restore_fn = loader.restore_predictions

        evaluator.evaluate(model, lkey, x_test, y_test, masks, restore_fn)
        return evaluator.load_results(lkey) or {}

    # ── Caching individual results ────────────────────────────────────────────

    def _cache_path(self, dataset: str, clip: float, tile: int) -> Path:
        return (
            RESULTS_DIR / dataset / "clahe_tuning"
            / f"{_dir_tag(clip, tile)}_cache.json"
        )

    def _cache_result(self, dataset: str, clip: float, tile: int, metrics: Dict) -> None:
        path = self._cache_path(dataset, clip, tile)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump({"clip": clip, "tile": tile, **metrics}, f, indent=2)

    def _load_cached(self, dataset: str, clip: float, tile: int) -> Optional[Dict]:
        path = self._cache_path(dataset, clip, tile)
        if not path.exists():
            return None
        with open(path) as f:
            data = json.load(f)
        # Strip the clip/tile fields so caller gets pure metrics dict
        return {k: v for k, v in data.items() if k not in ("clip", "tile")}

    # ── Best-combination selection (multi-metric composite score) ────────────

    @staticmethod
    def _composite_score(metrics: Dict, reference: List[Dict]) -> float:
        """Weighted composite score normalised against a reference set.

        Each metric is min-max normalised within `reference` so value ranges
        do not skew the result.  Lower-is-better metrics are inverted after
        normalisation.  Returns a value in [0, 1].
        """
        score = 0.0
        for metric, weight in SCORING_WEIGHTS.items():
            vals = [m.get(metric, 0.0) for m in reference if m]
            lo, hi = min(vals), max(vals)
            val  = metrics.get(metric, 0.0)
            norm = (val - lo) / (hi - lo) if hi > lo else 0.5
            if metric in _LOWER_IS_BETTER:
                norm = 1.0 - norm
            score += weight * norm
        return score

    def _avg_composite(
        self,
        all_results: Dict[str, Dict[str, Dict]],
        clip: float,
        tile: int,
    ) -> float:
        """Average composite score across both datasets for one (clip, tile)."""
        tag    = _tag(clip, tile)
        scores = []
        for dataset in DATASETS:
            d_data = all_results.get(dataset, {})
            target = d_data.get(tag)
            if not target:
                continue
            ref = [v for v in d_data.values() if v]
            if not ref:
                continue
            scores.append(self._composite_score(target, ref))
        return sum(scores) / len(scores) if scores else 0.0

    def _all_composite_scores(
        self, all_results: Dict[str, Dict[str, Dict]]
    ) -> Dict[str, Dict[str, float]]:
        """Compute composite scores for every (dataset, tag) combination."""
        out: Dict[str, Dict[str, float]] = {}
        for dataset in DATASETS:
            d_data = all_results.get(dataset, {})
            ref    = [v for v in d_data.values() if v]
            out[dataset] = {
                tag: self._composite_score(m, ref)
                for tag, m in d_data.items()
                if m and ref
            }
        return out

    def _pick_best_clip(
        self,
        all_results: Dict[str, Dict[str, Dict]],
        clips: List[float],
        tile: int,
    ) -> float:
        return max(
            clips,
            key=lambda c: self._avg_composite(all_results, c, tile),
        )

    def _pick_best_overall(
        self,
        all_results: Dict[str, Dict[str, Dict]],
        step_a_best_clip: float,
    ) -> Tuple[float, int]:
        candidates = [(c, STEP_A_TILE) for c in STEP_A_CLIPS]
        candidates += [(step_a_best_clip, t) for t in STEP_B_TILES]
        return max(
            candidates,
            key=lambda ct: self._avg_composite(all_results, ct[0], ct[1]),
        )

    # ── Persistence ───────────────────────────────────────────────────────────

    def _save_combined(
        self,
        all_results: Dict[str, Dict[str, Dict]],
        best_clip: float,
        best_tile: int,
        step_a_best_clip: float,
    ) -> None:
        scores = self._all_composite_scores(all_results)
        out = {
            "step_a_best_clip": step_a_best_clip,
            "recommendation": {
                "clipLimit":    best_clip,
                "tileGridSize": best_tile,
            },
            "scoring_weights": SCORING_WEIGHTS,
            "composite_scores": {
                dataset: {
                    tag: round(s, 4)
                    for tag, s in d_scores.items()
                }
                for dataset, d_scores in scores.items()
            },
            "search_space": {
                "step_a": {"clipLimit": STEP_A_CLIPS, "tileGridSize": STEP_A_TILE},
                "step_b": {"clipLimit": step_a_best_clip, "tileGridSize": STEP_B_TILES},
            },
            "results": {
                dataset: {tag: metrics for tag, metrics in data.items()}
                for dataset, data in all_results.items()
            },
        }
        path = self.out_dir / "clahe_tuning_results.json"
        with open(path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\n  Hasil lengkap tersimpan: {path}")

    # ── Plotting ──────────────────────────────────────────────────────────────

    def _plot(
        self,
        dataset: str,
        results: Dict[str, Dict],
        best_clip: float,
        best_tile: int,
    ) -> None:
        if not results:
            return

        tags      = list(results.keys())
        n_metrics = len(_METRIC_KEYS)
        n_cols    = 4
        n_rows    = (n_metrics + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
        axes_flat = axes.flatten()

        cmap      = plt.cm.tab10(np.linspace(0, 0.9, len(tags)))
        best_tag  = _tag(best_clip, best_tile)

        # Short x-tick labels: "c=1.5\nt=8"
        xlabels = [
            t.replace("clip", "c=").replace("_tile", "\nt=")
            for t in tags
        ]

        for ax_i, metric in enumerate(_METRIC_KEYS):
            ax   = axes_flat[ax_i]
            vals = [results.get(t, {}).get(metric, float('nan')) for t in tags]

            bar_colors = [
                '#d62728' if t == best_tag else tuple(cmap[i])
                for i, t in enumerate(tags)
            ]
            bars = ax.bar(
                range(len(tags)), vals,
                color=bar_colors, edgecolor='white', linewidth=0.5,
            )

            ax.set_xticks(range(len(tags)))
            ax.set_xticklabels(xlabels, fontsize=7, rotation=30, ha='right')

            is_lower = metric in _LOWER_IS_BETTER
            unit     = "Count (avg/img)" if is_lower else "%"
            title    = f"{metric}\n(↓ lower is better)" if is_lower else metric
            ax.set_title(title, fontsize=9, fontweight='bold')
            ax.set_ylabel(unit, fontsize=8)

            # Zoom y-axis so small differences are visible
            valid = [v for v in vals if not np.isnan(v)]
            if valid:
                lo, hi = min(valid), max(valid)
                pad = (hi - lo) * 0.35 if hi > lo else max(abs(hi) * 0.05, 0.5)
                ax.set_ylim(max(0, lo - pad), hi + pad)

            # Value annotations
            for bar, v in zip(bars, vals):
                if not np.isnan(v):
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height(),
                        f"{v:.2f}",
                        ha='center', va='bottom', fontsize=7,
                    )

        for ax in axes_flat[n_metrics:]:
            ax.set_visible(False)

        fig.suptitle(
            f"CLAHE Parameter Tuning — {dataset.upper()}\n"
            f"Best: clipLimit={best_clip}, tileGridSize={best_tile}  "
            f"(merah = kombinasi terpilih)",
            fontsize=12, fontweight='bold',
        )
        fig.tight_layout()

        path = self.out_dir / f"clahe_tuning_{dataset}.png"
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Chart tersimpan: {path}")

    # ── Terminal summary ──────────────────────────────────────────────────────

    @staticmethod
    def _print_brief(dataset: str, metrics: Dict) -> None:
        f1  = metrics.get('f1',           float('nan'))
        auc = metrics.get('auc',          float('nan'))
        b0  = metrics.get('betti0_error', float('nan'))
        print(
            f"    [{dataset.upper()}]  "
            f"F1={f1:.2f}%  AUC={auc:.2f}%  β0 Err={b0:.2f}"
        )

    def _print_summary(
        self,
        all_results: Dict[str, Dict[str, Dict]],
        best_clip: float,
        best_tile: int,
        step_a_best_clip: float,
    ) -> None:
        best_tag = _tag(best_clip, best_tile)
        scores   = self._all_composite_scores(all_results)

        print("\n" + "=" * 62)
        print("  CLAHE PARAMETER TUNING — HASIL AKHIR")
        print("=" * 62)
        print(f"  Scoring : weighted composite  "
              f"(F1×{SCORING_WEIGHTS['f1']:.0%}  "
              f"AUC×{SCORING_WEIGHTS['auc']:.0%}  "
              f"Sens×{SCORING_WEIGHTS['sensitivity']:.0%}  "
              f"β0×{SCORING_WEIGHTS['betti0_error']:.0%})")
        print(f"  Step A best clipLimit : {step_a_best_clip}  "
              f"(tileGridSize={STEP_A_TILE})")
        print(f"  Rekomendasi akhir     : clipLimit={best_clip}, "
              f"tileGridSize={best_tile}")

        # Composite score ranking table
        all_tags = sorted(
            {t for d in DATASETS for t in all_results.get(d, {})},
        )
        if all_tags:
            print()
            print(f"  {'Kombinasi':<18}  "
                  f"{'Score DRIVE':>11}  {'Score STARE':>11}  {'Score Avg':>9}")
            print("  " + "-" * 56)
            for tag in all_tags:
                sd = scores.get("drive", {}).get(tag, float('nan'))
                ss = scores.get("stare", {}).get(tag, float('nan'))
                avg = (sd + ss) / 2 if not (np.isnan(sd) or np.isnan(ss)) else float('nan')
                marker = "  ← best" if tag == best_tag else ""
                print(
                    f"  {tag:<18}  {sd:>11.4f}  {ss:>11.4f}  {avg:>9.4f}{marker}"
                )

        print()
        print("  Metrik terpilih pada kombinasi terbaik:")
        for dataset in DATASETS:
            m = all_results[dataset].get(best_tag, {})
            print(
                f"  [{dataset.upper()}]  "
                f"F1={m.get('f1', float('nan')):.2f}%  "
                f"Sensitivity={m.get('sensitivity', float('nan')):.2f}%  "
                f"AUC={m.get('auc', float('nan')):.2f}%  "
                f"β0 Err={m.get('betti0_error', float('nan')):.2f}"
            )
        print()
        print(f"  Gunakan parameter ini pada eksperimen utama:")
        print(f"    clahe_clip_limit = {best_clip}")
        print(f"    clahe_tile_grid  = {best_tile}")
        print("=" * 62)
