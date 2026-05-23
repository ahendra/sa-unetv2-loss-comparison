import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from config import LOSS_FUNCTIONS, RESULTS_DIR
from .palette import LOSS_COLORS, METRIC_COLORS, METRIC_LABELS


_PRIMARY_METRICS = ["accuracy", "f1", "sensitivity", "specificity", "auc", "mcc", "jaccard"]
_ALL_METRICS     = _PRIMARY_METRICS + ["cldice", "betti0_error", "betti1_error"]
_LOWER_IS_BETTER = {"betti0_error", "betti1_error"}

# Category mapping for Section 4.6.2
_LOSS_CATEGORIES = {
    "bce_mcc":   "Pixel-wise",
    "dice":      "Overlap",
    "focal":     "Imbalance-aware",
    "cldice":    "Topology-aware",
    "dice_ssim": "Hybrid (Area+Texture)",
    "bce_ssim":  "Hybrid (BCE+Texture)",
}


class ComparisonReporter:
    """Generate all comparison artifacts for Section 4.6.

    Outputs per dataset:
      - radar_chart_<dataset>.png   (spider chart, 6 metrics × 6 losses)
      - bar_chart_<dataset>.png     (grouped bar chart per metric)
      - ranking_table_<dataset>.json (ranked loss functions per metric)

    Single Responsibility: reads existing *_results.json files and produces
    visual + tabular comparisons. Does not perform training or inference.
    """

    def __init__(self, output_dir: Path):
        self._out_dir = output_dir

    def generate_all(self, dataset: str = "drive") -> Dict[str, Path]:
        """Load results and generate all comparison outputs.

        Args:
            dataset: 'drive' or 'stare'

        Returns:
            Dict mapping output type to saved Path.
        """
        results = self._load_results(dataset)
        if not results:
            print(f"  [WARN] Tidak ada hasil evaluasi untuk dataset '{dataset}'.")
            return {}

        self._out_dir.mkdir(parents=True, exist_ok=True)
        paths: Dict[str, Path] = {}
        paths["radar"]     = self._plot_radar(results, dataset)
        paths["bar"]       = self._plot_bar(results, dataset)
        paths["confusion"] = self._plot_confusion_matrices(results, dataset)
        paths["ranking"]   = self._save_ranking(results, dataset)
        paths["wilcoxon"]  = self._run_wilcoxon_test(results, dataset)
        return paths

    # ── Loaders ──────────────────────────────────────────────────────────────

    def _load_results(self, dataset: str) -> Dict[str, Dict]:
        results = {}
        for loss_key in LOSS_FUNCTIONS:
            path = RESULTS_DIR / dataset / f"{loss_key}_results.json"
            if path.exists():
                with open(path) as f:
                    results[loss_key] = json.load(f)
        return results

    # ── Radar chart ───────────────────────────────────────────────────────────

    def _plot_radar(self, results: Dict, dataset: str) -> Optional[Path]:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print("  [WARN] matplotlib tidak terinstall, skip radar chart.")
            return None

        # Overlap + topology metrics; Betti errors inverted (lower raw = outer ring)
        metrics = _ALL_METRICS
        n_m     = len(metrics)
        angles  = np.linspace(0, 2 * np.pi, n_m, endpoint=False).tolist()
        angles += angles[:1]

        _MARKERS = ['o', 's', '^', 'D', 'v', 'P']

        # Per-metric min-max normalisation → [_LO, _HI]
        # Betti errors inverted so that outer ring always = better for all metrics.
        _LO, _HI = 0.08, 0.92
        metric_stats: Dict[str, tuple] = {}
        for m in metrics:
            vals_m = [results[k].get(m, 0) for k in LOSS_FUNCTIONS if k in results]
            metric_stats[m] = (min(vals_m), max(vals_m))

        def _norm(val_raw: float, m: str) -> float:
            lo, hi = metric_stats[m]
            rng = (hi - lo) if (hi - lo) > 1e-9 else 1e-9
            t   = (val_raw - lo) / rng          # 0.0 (lo) → 1.0 (hi)
            if m in _LOWER_IS_BETTER:
                t = 1.0 - t                      # invert: low raw = outer ring
            return _LO + t * (_HI - _LO)

        # Spoke labels (from shared METRIC_LABELS, Betti use multi-line versions)
        _spoke_label = {
            **METRIC_LABELS,
            "betti0_error": "β₀ Error\n(↓ lebih kecil\nlebih baik)",
            "betti1_error": "β₁ Error\n(↓ lebih kecil\nlebih baik)",
        }
        spoke_labels = [_spoke_label.get(m, m.upper()) for m in metrics]

        fig, ax = plt.subplots(figsize=(11, 12), subplot_kw=dict(polar=True))
        fig.patch.set_facecolor("#ffffff")
        ax.set_facecolor("#f8f9fa")

        for idx, (loss_key, loss_label) in enumerate(LOSS_FUNCTIONS.items()):
            if loss_key not in results:
                continue
            color     = LOSS_COLORS.get(loss_key, "#aaaaaa")
            marker    = _MARKERS[idx % len(_MARKERS)]
            norm_vals = [_norm(results[loss_key].get(m, 0), m) for m in metrics]
            norm_vals += norm_vals[:1]
            ax.plot(angles, norm_vals,
                    color=color, lw=2.2, linestyle="-",
                    marker=marker, markersize=7, markerfacecolor=color,
                    markeredgecolor="white", markeredgewidth=0.8,
                    label=loss_label.replace(" (Baseline)", ""),
                    zorder=3)
            ax.fill(angles, norm_vals, color=color, alpha=0.05, zorder=2)

        # Radial grid rings
        r_ticks = np.linspace(_LO, _HI, 5)
        ax.set_ylim(0.0, 1.0)
        ax.set_yticks(r_ticks.tolist())
        ax.set_yticklabels(["Terburuk", "", "", "", "Terbaik"],
                           fontsize=7, color="#888888")
        ax.yaxis.set_tick_params(pad=6)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(spoke_labels, fontsize=9, color="#222222")

        # Spoke grid: slightly darker than default
        ax.grid(color="#cccccc", linestyle="-", linewidth=0.6, alpha=0.8)
        ax.spines["polar"].set_color("#cccccc")

        ax.set_title(
            f"Loss Function Comparison — {dataset.upper()}",
            fontsize=12, fontweight="bold", pad=30, color="#111111",
        )

        handles, labels = ax.get_legend_handles_labels()
        fig.legend(
            handles, labels,
            loc="lower center", ncol=3,
            bbox_to_anchor=(0.5, 0.01),
            fontsize=9.5, framealpha=0.95,
            edgecolor="#dddddd",
        )

        path = self._out_dir / f"radar_chart_{dataset}.png"
        fig.savefig(str(path), dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f"  Radar chart: {path}")
        return path

    # ── Bar chart ─────────────────────────────────────────────────────────────

    def _plot_bar(self, results: Dict, dataset: str) -> Optional[Path]:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            return None

        # Overlap + topology metrics
        metrics = _ALL_METRICS
        _y_axis_label = {
            "betti0_error": "Error Count",
            "betti1_error": "Error Count",
        }

        loss_keys   = [k for k in LOSS_FUNCTIONS if k in results]
        loss_labels = [LOSS_FUNCTIONS[k].replace(" (Baseline)", "") for k in loss_keys]
        n_losses = len(loss_keys)
        colors   = [LOSS_COLORS.get(k, "#aaaaaa") for k in loss_keys]

        n_cols = 3
        n_rows = (len(metrics) + n_cols - 1) // n_cols
        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(16, 6 * n_rows),
            gridspec_kw={"hspace": 0.80, "wspace": 0.40},
        )
        fig.suptitle(
            f"Loss Function Comparison — {dataset.upper()}",
            fontsize=12, fontweight="bold",
        )
        axes_flat = np.array(axes).flatten()

        for m_idx, mkey in enumerate(metrics):
            ax     = axes_flat[m_idx]
            mlabel = METRIC_LABELS.get(mkey, mkey.upper())
            mcolor = METRIC_COLORS.get(mkey, "#111111")
            vals   = [results[k].get(mkey, 0) for k in loss_keys]
            lower_better = mkey in _LOWER_IS_BETTER

            v_min = min(vals)
            v_max = max(vals)
            span  = (v_max - v_min) if (v_max - v_min) > 1e-9 else 0.02

            y_min = max(0.0, v_min - span * 0.4)
            y_max = v_max + span * 0.6
            if not lower_better:
                y_max = min(100.0, y_max)
            if (y_max - y_min) < 0.01:
                y_min = max(0.0, v_min - 0.05)
                y_max = v_max + 0.05
            label_offset = (y_max - y_min) * 0.02

            for i, (k, color) in enumerate(zip(loss_keys, colors)):
                v = results[k].get(mkey, 0)
                ax.bar(i, v, width=0.65, color=color, alpha=0.85,
                       edgecolor="white", label=loss_labels[i])
                ax.text(i, v + label_offset, f"{v:.2f}",
                        ha="center", va="bottom", fontsize=7, rotation=45)

            ax.set_xticks(range(n_losses))
            ax.set_xticklabels(loss_labels, rotation=35, ha="right", fontsize=8)
            ax.set_ylabel(_y_axis_label.get(mkey, "Score (%)"), fontsize=9)
            ax.set_title(mlabel, fontsize=10, fontweight="bold", color=mcolor)
            ax.set_xlim(-0.6, n_losses - 0.4)
            ax.set_ylim(y_min, y_max)
            ax.grid(axis="y", alpha=0.3)

        # Hide unused panels
        for idx in range(len(metrics), len(axes_flat)):
            axes_flat[idx].axis("off")

        # Single figure-level legend below all subplots
        handles, labels = axes_flat[0].get_legend_handles_labels()
        fig.legend(
            handles, labels,
            loc="lower center", ncol=min(n_losses, 3),
            bbox_to_anchor=(0.5, -0.02),
            fontsize=9, framealpha=0.9,
        )

        path = self._out_dir / f"bar_chart_{dataset}.png"
        fig.savefig(str(path), dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Bar chart: {path}")
        return path

    # ── Confusion matrix grid ─────────────────────────────────────────────────

    def _plot_confusion_matrices(self, results: Dict, dataset: str) -> Optional[Path]:
        """Plot one 2×2 confusion matrix per loss function using actual pixel counts.

        Pixel counts (TP, TN, FP, FN) are read from *_results.json.
        If a file predates this feature (no count keys), falls back to
        deriving approximate rates from Sensitivity / Specificity.

        Layout — rows = Actual class, cols = Predicted class:
          [TN  FP]   actual = Background (non-vessel)
          [FN  TP]   actual = Vessel

        Colour scheme matches segmentation_grid Peta Kesalahan:
          TP = Hijau    FP = Merah    FN = Biru    TN = Hijau muda
        """
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print("  [WARN] matplotlib tidak terinstall, skip confusion matrix.")
            return None

        loss_keys   = [k for k in LOSS_FUNCTIONS if k in results]
        loss_labels = [LOSS_FUNCTIONS[k].replace(" (Baseline)", "") for k in loss_keys]
        n = len(loss_keys)

        n_cols = 3
        n_rows = (n + n_cols - 1) // n_cols
        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(n_cols * 4.8, n_rows * 4.6),
            gridspec_kw={"hspace": 0.60, "wspace": 0.40},
        )
        axes_flat = np.array(axes).flatten()

        # Cell colours (RGB 0–1) — consistent with visualization_reporter
        _CELL_COLOR = {
            "TN": np.array([0.55, 0.88, 0.55]),   # hijau muda
            "FP": np.array([0.92, 0.40, 0.40]),   # merah
            "FN": np.array([0.40, 0.40, 0.92]),   # biru
            "TP": np.array([0.10, 0.68, 0.10]),   # hijau tua
        }
        # [row][col] mapping: row=Actual, col=Predicted
        _CELL_KEY = [["TN", "FP"],   # Actual = Background
                     ["FN", "TP"]]   # Actual = Vessel

        for idx, (loss_key, label) in enumerate(zip(loss_keys, loss_labels)):
            ax = axes_flat[idx]
            m  = results[loss_key]

            # --- Retrieve per-image average pixel counts -----------------
            has_counts = all(k in m for k in
                             ("tp_count", "tn_count", "fp_count", "fn_count"))
            n_imgs = max(int(m.get("num_images", 1)), 1)

            if has_counts:
                # Opsi B: average pixel count per image
                tp = m["tp_count"] / n_imgs
                tn = m["tn_count"] / n_imgs
                fp = m["fp_count"] / n_imgs
                fn = m["fn_count"] / n_imgs
            else:
                # Fallback: derive approximate rates from averaged metrics
                # (shown as float rates, flagged for re-evaluation)
                tpr = m.get("sensitivity", 0) / 100.0
                tnr = m.get("specificity", 0) / 100.0
                tp, tn, fp, fn = tpr, tnr, 1.0 - tnr, 1.0 - tpr

            total = tp + tn + fp + fn if (tp + tn + fp + fn) > 0 else 1

            # counts[row][col]: row=Actual, col=Predicted
            counts = [[tn, fp],
                      [fn, tp]]

            # --- Cell background: intensity ∝ count / row maximum -------
            row_maxes = [max(counts[r][0], counts[r][1]) for r in range(2)]
            rgb_img = np.ones((2, 2, 3), dtype=np.float64)
            for r in range(2):
                for c in range(2):
                    v    = counts[r][c] / (row_maxes[r] + 1e-9)
                    base = _CELL_COLOR[_CELL_KEY[r][c]]
                    rgb_img[r, c] = 1.0 - v * (1.0 - base)

            ax.imshow(rgb_img, interpolation="nearest", aspect="auto",
                      extent=[-0.5, 1.5, 1.5, -0.5])
            ax.axhline(0.5, color="white", lw=3.0)
            ax.axvline(0.5, color="white", lw=3.0)

            # --- Cell text: label / avg pixel count / percentage ---------
            for r in range(2):
                for c in range(2):
                    cnt      = counts[r][c]
                    cell_key = _CELL_KEY[r][c]
                    pct      = cnt / total * 100
                    bg_val   = cnt / (row_maxes[r] + 1e-9)
                    txt_col  = "white" if bg_val > 0.55 else "black"

                    if has_counts:
                        count_str = f"{cnt:,.0f} px"
                        pct_str   = f"({pct:.2f}%)"
                    else:
                        count_str = f"{cnt * 100:.2f}%"
                        pct_str   = "(~estimasi)"

                    ax.text(c, r - 0.18, cell_key,
                            ha="center", va="center",
                            fontsize=12, fontweight="bold", color=txt_col)
                    ax.text(c, r + 0.10, count_str,
                            ha="center", va="center",
                            fontsize=10, fontweight="bold", color=txt_col)
                    ax.text(c, r + 0.35, pct_str,
                            ha="center", va="center",
                            fontsize=8, color=txt_col)

            ax.set_xticks([0, 1])
            ax.set_xticklabels(["Prediksi\nBackground", "Prediksi\nVessel"],
                                fontsize=9)
            ax.set_yticks([0, 1])
            ax.set_yticklabels(["Aktual\nBackground", "Aktual\nVessel"],
                                fontsize=9)
            ax.tick_params(length=0)
            ax.set_title(label, fontsize=10, fontweight="bold", pad=10)
            if not has_counts:
                ax.set_xlabel("⚠ Re-run evaluasi untuk pixel counts",
                              fontsize=7, color="gray")

        # Hide unused panels
        for idx in range(n, len(axes_flat)):
            axes_flat[idx].axis("off")

        subtitle = (
            "Rata-rata jumlah pixel per gambar test (total ÷ jumlah gambar)\n"
            "Persentase relatif terhadap total pixel rata-rata per gambar"
        )
        fig.suptitle(
            f"Confusion Matrix — {dataset.upper()}\n{subtitle}",
            fontsize=11, fontweight="bold",
        )

        path = self._out_dir / f"confusion_matrix_{dataset}.png"
        fig.savefig(str(path), dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Confusion matrix: {path}")
        return path

    # ── Wilcoxon signed-rank test (Sub-bab 4.7.5) ────────────────────────────

    def _run_wilcoxon_test(self, results: Dict, dataset: str) -> Optional[Path]:
        """Pairwise Wilcoxon signed-rank test across all loss function pairs.

        Requires per_image_metrics in each *_results.json (saved by evaluator
        when evaluate() is run after the evaluator update). If data is missing,
        prints a warning and returns None.

        Outputs:
          - wilcoxon_<dataset>.json  — full p-value matrix per metric
          - wilcoxon_<dataset>.txt   — human-readable summary table
          - wilcoxon_heatmap_<dataset>.png — p-value heatmap (F1)
        """
        try:
            from scipy import stats as scipy_stats
        except ImportError:
            print("  [WARN] scipy tidak terinstall, skip Wilcoxon test.")
            return None

        # Overlap metrics (higher=better) + topology metrics (lower=better)
        _TEST_METRICS        = ["accuracy", "f1", "sensitivity", "specificity", "auc",
                                "jaccard", "cldice", "betti0_error", "betti1_error"]
        _LOWER_IS_BETTER_SET = {"betti0_error", "betti1_error"}

        # Only include losses that have per_image_metrics
        loss_keys = [
            k for k in LOSS_FUNCTIONS
            if k in results and "per_image_metrics" in results[k]
        ]
        if len(loss_keys) < 2:
            print(
                "  [INFO] Wilcoxon test: per_image_metrics belum tersedia. "
                "Re-run evaluasi untuk mendapatkan data per-gambar."
            )
            return None

        n = len(loss_keys)
        wilcoxon_output: Dict = {"dataset": dataset, "metrics": {}}

        for metric in _TEST_METRICS:
            pairs: Dict = {}
            for k1 in loss_keys:
                for k2 in loss_keys:
                    if k1 >= k2:
                        continue
                    x = [m[metric] for m in results[k1]["per_image_metrics"]
                         if metric in m]
                    y = [m[metric] for m in results[k2]["per_image_metrics"]
                         if metric in m]
                    if len(x) != len(y) or len(x) < 5:
                        continue
                    diffs = [xi - yi for xi, yi in zip(x, y)]
                    if all(d == 0 for d in diffs):
                        # Semua gambar identik — topologi sempurna di kedua loss
                        pairs[f"{k1}_vs_{k2}"] = {
                            "p_value": None,
                            "significant_005": False,
                            "note": "all_differences_zero",
                        }
                        continue
                    try:
                        stat, p = scipy_stats.wilcoxon(x, y)
                        pairs[f"{k1}_vs_{k2}"] = {
                            "statistic": round(float(stat), 4),
                            "p_value":   round(float(p), 6),
                            "significant_005": bool(p < 0.05),
                            "significant_001": bool(p < 0.01),
                        }
                    except Exception as exc:
                        pairs[f"{k1}_vs_{k2}"] = {
                            "p_value": None,
                            "significant_005": False,
                            "note": str(exc),
                        }
            wilcoxon_output["metrics"][metric] = pairs

        # ── Save JSON ──────────────────────────────────────────────────────────
        json_path = self._out_dir / f"wilcoxon_{dataset}.json"
        with open(json_path, "w") as f:
            json.dump(wilcoxon_output, f, indent=2)

        # ── Save TXT summary ───────────────────────────────────────────────────
        txt_path = self._out_dir / f"wilcoxon_{dataset}.txt"
        _metric_note = {
            "accuracy":     "Overlap  | higher=better",
            "f1":           "Overlap  | higher=better",
            "sensitivity":  "Overlap  | higher=better",
            "specificity":  "Overlap  | higher=better",
            "auc":          "Overlap  | higher=better",
            "jaccard":      "Overlap  | higher=better",
            "cldice":       "Topology | higher=better",
            "betti0_error": "Topology | lower=better  (komponen putus)",
            "betti1_error": "Topology | lower=better  (loop palsu)",
        }
        lines = [
            f"Wilcoxon Signed-Rank Test — {dataset.upper()}",
            f"{'=' * 65}",
            "Hipotesis nol (H0): tidak ada perbedaan distribusi antar fungsi loss.",
            "Ditolak jika p-value < 0.05 (*) atau < 0.01 (**).",
            "Catatan: Betti errors bernilai integer ≥ 0; 'all_differences_zero'",
            "         berarti kedua loss mencapai topologi sempurna di semua gambar.",
            "",
        ]
        for metric in _TEST_METRICS:
            note = _metric_note.get(metric, "")
            lines.append(f"Metrik: {metric.upper()}  [{note}]")
            lines.append(f"  {'Pasangan':<30} {'p-value':>10}  {'Keterangan'}")
            lines.append(f"  {'-'*65}")
            pairs = wilcoxon_output["metrics"].get(metric, {})
            for pair_key, v in pairs.items():
                p    = v.get("p_value")
                note = v.get("note", "")
                if note == "all_differences_zero":
                    sig   = "topologi identik di semua gambar"
                    p_str = "–"
                elif p is None:
                    sig   = f"error: {note}"
                    p_str = "N/A"
                elif p < 0.01:
                    sig   = "** signifikan (p<0.01)"
                elif p < 0.05:
                    sig   = "*  signifikan (p<0.05)"
                else:
                    sig   = "tidak signifikan"
                    p_str = f"{p:.6f}"
                if p is not None and note not in ("all_differences_zero",):
                    p_str = f"{p:.6f}"
                lines.append(f"  {pair_key:<30} {p_str:>10}  {sig}")
            lines.append("")
        txt_path.write_text("\n".join(lines), encoding="utf-8")

        # ── Save heatmap PNG (F1 overlap + clDice topology, side-by-side) ────────
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            idx_map     = {k: i for i, k in enumerate(loss_keys)}
            short_labels = [LOSS_FUNCTIONS[k].replace(" (Baseline)", "")
                            for k in loss_keys]

            def _build_matrix(metric_key: str) -> np.ndarray:
                mat = np.full((n, n), np.nan)
                np.fill_diagonal(mat, 1.0)
                for pair_key, v in wilcoxon_output["metrics"].get(metric_key, {}).items():
                    k1, k2 = pair_key.split("_vs_")
                    p = v.get("p_value")
                    if p is not None and k1 in idx_map and k2 in idx_map:
                        mat[idx_map[k1], idx_map[k2]] = p
                        mat[idx_map[k2], idx_map[k1]] = p
                return mat

            def _annotate(ax_, mat_):
                for i in range(n):
                    for j in range(n):
                        val = mat_[i, j]
                        if np.isnan(val) or i == j:
                            ax_.text(j, i, "–", ha="center", va="center",
                                     fontsize=8, color="gray")
                        elif val < 0.01:
                            ax_.text(j, i, f"{val:.4f}\n(**)", ha="center",
                                     va="center", fontsize=8, color="white")
                        elif val < 0.05:
                            ax_.text(j, i, f"{val:.4f}\n(*)", ha="center",
                                     va="center", fontsize=8, color="black")
                        else:
                            ax_.text(j, i, f"{val:.4f}", ha="center",
                                     va="center", fontsize=8, color="black")

            heatmap_specs = [
                ("f1",     "F1 Score (Overlap)"),
                ("cldice", "clDice (Topology)"),
            ]
            fig, axes = plt.subplots(1, 2, figsize=(16, 7),
                                     gridspec_kw={"wspace": 0.45})
            fig.suptitle(
                f"Wilcoxon Signed-Rank Test — {dataset.upper()}\n"
                f"Hijau = signifikan  |  * p<0.05   ** p<0.01",
                fontsize=11, fontweight="bold",
            )

            for ax, (mkey, mlabel) in zip(axes, heatmap_specs):
                mat = _build_matrix(mkey)
                im  = ax.imshow(mat, cmap="RdYlGn_r", vmin=0.0, vmax=0.10,
                                aspect="auto")
                plt.colorbar(im, ax=ax, label="p-value", shrink=0.85)
                ax.set_xticks(range(n)); ax.set_yticks(range(n))
                ax.set_xticklabels(short_labels, rotation=30, ha="right", fontsize=9)
                ax.set_yticklabels(short_labels, fontsize=9)
                ax.set_title(mlabel, fontsize=10, fontweight="bold")
                _annotate(ax, mat)

            heatmap_path = self._out_dir / f"wilcoxon_heatmap_{dataset}.png"
            fig.savefig(str(heatmap_path), dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"  Wilcoxon heatmap: {heatmap_path}")
        except Exception as e:
            print(f"  [WARN] Gagal membuat heatmap Wilcoxon: {e}")

        print(f"  Wilcoxon JSON: {json_path}")
        print(f"  Wilcoxon TXT : {txt_path}")
        return json_path

    # ── Ranking table ─────────────────────────────────────────────────────────

    def _save_ranking(self, results: Dict, dataset: str) -> Path:
        ranking: Dict = {}
        for metric in _ALL_METRICS:
            reverse = metric not in _LOWER_IS_BETTER
            items = [
                (k, results[k].get(metric, float("nan")))
                for k in LOSS_FUNCTIONS
                if k in results
            ]
            items.sort(key=lambda x: (x[1] != x[1], x[1]),
                       reverse=reverse)
            ranking[metric] = [
                {
                    "rank":       i + 1,
                    "loss_key":   k,
                    "loss_label": LOSS_FUNCTIONS[k],
                    "category":   _LOSS_CATEGORIES.get(k, ""),
                    "value":      round(v, 2) if v == v else None,
                }
                for i, (k, v) in enumerate(items)
            ]

        path = self._out_dir / f"ranking_table_{dataset}.json"
        with open(path, "w") as f:
            json.dump(ranking, f, indent=2)
        print(f"  Ranking table: {path}")
        return path
