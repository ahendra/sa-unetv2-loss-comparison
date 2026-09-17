"""
Multi-seed analysis reporter — revised per R1–R8.

R1: Five-seed-averaged per-image values as unit of analysis for Wilcoxon.
R2: Matched-pairs rank-biserial r_rb = (R_plus − R_minus) / (R_plus + R_minus).
R3: All 15 pairwise comparisons C(6,2), not just vs baseline.
R4: Holm (1979) correction as PRIMARY; BH-FDR supplementary only.
R5: Two-sided Wilcoxon, zero_method="pratt" (unchanged).
R6: STARE N=4 — descriptive only, no significance testing.
R7: Pair by image_id; validate alignment across seeds and losses.
R8: All 5 seeds required; fail clearly if any result file is missing.

Output files
------------
  1. multiseed_summary.json     — mean ± SD per loss × metric (N seeds)
  2. per_image_averaged.json    — 5-seed averaged per-image values per loss
  3. pairwise_wilcoxon.json     — all pairs per metric, Holm-corrected (full data)
  4. pairwise_wilcoxon.csv      — manuscript-friendly CSV (full data)
  5. significance_wtl.txt/csv   — Win-Tie-Loss summary table per loss function
  6. significance_matrix.png    — colour-coded 7×7 significance matrix (F1, AUC, Sens, clDice)
  7. significant_pairs.txt      — compact list of only Holm-significant pairs + effect size
  8. boxplot_{metric}.png       — per-metric seed-distribution boxplots

Statistical design
------------------
- Unit of analysis  : per-image metrics averaged over ALL 5 seeds (R1).
  Same test image evaluated by each trained model → paired observations.
  N = 20 (DRIVE) gives adequate power; N = 4 (STARE) → descriptive only (R6).
- Test               : Wilcoxon signed-rank, two-sided, zero_method="pratt" (R5).
- Effect size        : matched-pairs rank-biserial r_rb (Cureton 1956; R2).
  r_rb = (R_plus − R_minus) / (R_plus + R_minus),  range [−1, +1].
  Positive = lossA better than lossB.
- Multiple testing   : Holm (1979) step-down per metric family of 15 pairs (R4).
  BH-FDR stored as supplementary.

References
----------
Wilcoxon (1945); Pratt (1959); Holm (1979, Scand. J. Statist.);
Cureton (1956); Kerby (2014); Benjamini & Hochberg (1995).
Retinal seg papers: Shit et al. (2021), Kirchhoff et al. (2024).
"""

import itertools
import json
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np

from config import DriveConfig, StareConfig, LOSS_FUNCTIONS, RESULTS_DIR


_REQUIRED_SEEDS       = [42, 123, 456, 789, 2026]
_OVERLAP_METRICS      = ["accuracy", "sensitivity", "specificity", "auc",
                         "mcc", "f1", "jaccard", "cldice"]
_TOPO_METRICS         = ["betti0_error", "betti1_error"]
_ALL_METRICS          = _OVERLAP_METRICS + _TOPO_METRICS
_LOWER_IS_BETTER      = {"betti0_error", "betti1_error"}
_MIN_N_FOR_WILCOXON   = 10   # below this, skip inferential testing (R6)

_METRIC_LABELS = {
    "accuracy":     "Accuracy (%)",
    "sensitivity":  "Sensitivity (%)",
    "specificity":  "Specificity (%)",
    "auc":          "AUC (%)",
    "mcc":          "MCC (%)",
    "f1":           "F1 (%)",
    "jaccard":      "Jaccard (%)",
    "cldice":       "clDice (%)",
    "betti0_error": "β0 Error",
    "betti1_error": "β1 Error",
}

_SIG_MARKERS = [(0.001, "***"), (0.01, "**"), (0.05, "*")]

# Metrics shown in the significance matrix PNG — all 10, arranged in 2 rows
# Row 0 (overlap): f1, auc, sensitivity, specificity, accuracy
# Row 1 (topology+): mcc, jaccard, cldice, betti0_error, betti1_error
_PRIMARY_VIZ_METRICS = [
    "f1", "auc", "sensitivity", "specificity", "accuracy",
    "mcc", "jaccard", "cldice", "betti0_error", "betti1_error",
]
_VIZ_LAYOUT = (2, 5)   # (n_rows, n_cols) for the matrix PNG

# Effect-size thresholds for rank-biserial r_rb (Kerby 2014)
_EFFECT_CATS = [(0.50, "Large"), (0.30, "Medium"), (0.10, "Small"), (0.00, "Negligible")]


def _sig_marker(p: Optional[float]) -> str:
    if p is None:
        return "n/a"
    for threshold, marker in _SIG_MARKERS:
        if p < threshold:
            return marker
    return "ns"


def _effect_label(r_rb: float) -> str:
    """Kerby (2014) effect-size category for matched-pairs rank-biserial r_rb."""
    for thresh, label in _EFFECT_CATS:
        if abs(r_rb) >= thresh:
            return label
    return "Negligible"


def _holm_correction(p_values: List[float]) -> List[float]:
    """Holm (1979) step-down correction. Returns adjusted p-values (same order as input)."""
    n = len(p_values)
    if n == 0:
        return []
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    adjusted = [0.0] * n
    running_max = 0.0
    for step, (orig_i, p) in enumerate(indexed):
        adj = p * (n - step)
        running_max = max(adj, running_max)
        adjusted[orig_i] = min(running_max, 1.0)
    return adjusted


def _bh_fdr(p_values: List[float]) -> List[float]:
    """Benjamini-Hochberg (1995) FDR — supplementary only. Same-order output."""
    n = len(p_values)
    if n == 0:
        return []
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    adjusted = [0.0] * n
    prev = 1.0
    for rank, (orig_i, p) in enumerate(reversed(indexed), 1):
        adj = p * n / (n - rank + 1)
        adj = min(adj, prev)
        prev = adj
        adjusted[orig_i] = min(adj, 1.0)
    return adjusted


def _matched_pairs_rank_biserial(a: np.ndarray, b: np.ndarray) -> float:
    """Matched-pairs rank-biserial r_rb (Cureton 1956 / Kerby 2014).

    Uses Pratt's variant: rank ALL |d_i| including zeros, then exclude
    zero-pairs from R_plus / R_minus sums.

    r_rb = (R_plus − R_minus) / (R_plus + R_minus),  ∈ [−1, +1].
    Positive → a > b on average.  Returns 0.0 if all differences are zero.
    """
    from scipy.stats import rankdata
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    abs_d = np.abs(d)
    ranks = rankdata(abs_d)            # rank |d_i| including zeros
    nonzero = abs_d > 0
    if not np.any(nonzero):
        return 0.0
    r_plus  = float(np.sum(ranks[nonzero & (d > 0)]))
    r_minus = float(np.sum(ranks[nonzero & (d < 0)]))
    denom = r_plus + r_minus
    if denom == 0.0:
        return 0.0
    return (r_plus - r_minus) / denom


class MultiSeedReporter:
    """
    Multi-seed statistical analysis reporter.

    Parameters
    ----------
    cfg   : DriveConfig or StareConfig
    out_dir : output directory for saved files
    seeds : list of seeds — MUST equal [42, 123, 456, 789, 2026] (R8)
    """

    def __init__(
        self,
        cfg: Union[DriveConfig, StareConfig],
        out_dir,
        seeds: List[int],
    ):
        self.cfg     = cfg
        self.out_dir = Path(out_dir)
        self.seeds   = list(seeds)
        self.dataset = cfg.name.lower()

    def generate(self) -> List[Path]:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        paths: List[Path] = []

        print(f"\n  {'═' * 60}")
        print(f"  Multi-Seed Analysis — {self.cfg.name}")
        print(f"  Seeds: {self.seeds}  |  Required: {_REQUIRED_SEEDS}")
        print(f"  {'═' * 60}")

        # R8: All seeds must be present before proceeding
        self._validate_all_seeds()

        # Step 1: Mean ± SD across seeds from aggregate results
        summary = self._compute_multiseed_summary()

        # Step 2: Five-seed-averaged per-image values (R1, R7)
        avg_per_image = self._compute_averaged_per_image()

        # Step 3: All 15 pairwise Wilcoxon tests (R2–R6)
        pairwise = self._run_pairwise_wilcoxon(avg_per_image)

        # Step 4: Print and save outputs
        self._print_summary_table(summary, pairwise)

        p = self._save_summary_json(summary)
        paths.append(p)
        print(f"\n  [1/4] Summary JSON    : {p}")

        p2 = self._save_per_image_json(avg_per_image)
        paths.append(p2)
        print(f"  [2/4] Per-image JSON  : {p2}")

        p3 = self._save_pairwise_json(pairwise)
        paths.append(p3)
        print(f"  [3/4] Pairwise JSON   : {p3}")

        p4 = self._save_pairwise_csv(pairwise)
        paths.append(p4)
        print(f"  [4/4] Pairwise CSV    : {p4}")

        plot_paths = self._plot_boxplots(summary)
        paths.extend(plot_paths)

        print(f"\n  ── Compact Significance Reports ──")
        sig_paths = self._generate_compact_significance(pairwise)
        paths.extend(sig_paths)

        return paths

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _result_path(self, loss_key: str, seed_tag: str = "") -> Path:
        tag = f"_{seed_tag}" if seed_tag else ""
        return RESULTS_DIR / self.dataset / f"{loss_key}{tag}_results.json"

    def _validate_all_seeds(self) -> None:
        """R8: Fail clearly listing every missing file before any computation."""
        missing = []
        for loss_key in LOSS_FUNCTIONS:
            for seed in self.seeds:
                path = self._result_path(loss_key, f"seed{seed}")
                if not path.exists():
                    missing.append(str(path.name))
        if not missing:
            return
        lines = [
            f"\n  [ERROR] {len(missing)} result file(s) missing.",
            f"  All {len(self.seeds)} seeds × {len(LOSS_FUNCTIONS)} loss functions",
            f"  must be evaluated before running multi-seed analysis.\n",
            f"  Missing files:",
        ]
        for fname in missing:
            lines.append(f"    • {fname}")
        lines += [
            "",
            f"  Run: Menu → Evaluasi → All Loss Functions for each seed:",
        ]
        for s in self.seeds:
            lines.append(f"    seed{s}")
        raise FileNotFoundError("\n".join(lines))

    # ── Mean ± SD across seeds ────────────────────────────────────────────────

    def _compute_multiseed_summary(self) -> Dict:
        """For each loss function, load per-seed aggregate JSONs → mean ± SD."""
        summary: Dict = {}
        for loss_key in LOSS_FUNCTIONS:
            seed_data: List[Dict] = []
            for seed in self.seeds:
                path = self._result_path(loss_key, f"seed{seed}")
                with open(path) as f:
                    d = json.load(f)
                d.pop("per_image_metrics", None)
                seed_data.append(d)

            per_metric: Dict = {}
            for m in _ALL_METRICS:
                vals = [d[m] for d in seed_data if m in d and d[m] is not None]
                if vals:
                    n = len(vals)
                    per_metric[m] = {
                        "mean": float(np.mean(vals)),
                        "std":  float(np.std(vals, ddof=1)) if n > 1 else 0.0,
                        "n":    n,
                        "raw":  [float(v) for v in vals],
                    }
            summary[loss_key] = per_metric
        return summary

    # ── Five-seed-averaged per-image values (R1, R7) ─────────────────────────

    def _compute_averaged_per_image(self) -> Dict:
        """
        For each (loss, image_id), average each metric over all seeds (R1).

        Validates image_id alignment across seeds and across losses (R7).

        Returns
        -------
        Dict[loss_key → Dict[image_id → Dict[metric → float]]]
        """
        per_loss: Dict[str, Dict[str, Dict[str, float]]] = {}

        for loss_key in LOSS_FUNCTIONS:
            seed_records: List[Dict[str, Dict]] = []
            for seed in self.seeds:
                path = self._result_path(loss_key, f"seed{seed}")
                with open(path) as f:
                    d = json.load(f)
                pim = d.get("per_image_metrics", [])
                if not pim:
                    raise ValueError(
                        f"per_image_metrics missing in {path.name}.\n"
                        f"Re-evaluate with the updated evaluator that stores image_id."
                    )
                by_id: Dict[str, Dict] = {}
                for entry in pim:
                    img_id = entry.get("image_id")
                    if img_id is None:
                        raise ValueError(
                            f"image_id missing in per_image_metrics of {path.name}.\n"
                            f"Re-evaluate using the updated evaluator (R7 fix)."
                        )
                    by_id[img_id] = {k: v for k, v in entry.items()
                                     if k != "image_id"}
                seed_records.append(by_id)

            # Validate identical image IDs across all seeds for this loss
            ref_ids = sorted(seed_records[0].keys())
            for si, sr in enumerate(seed_records[1:], 1):
                if sorted(sr.keys()) != ref_ids:
                    raise ValueError(
                        f"loss={loss_key}: image IDs differ between "
                        f"seed[0] and seed[{si}] (seed={self.seeds[si]}).\n"
                        f"Ensure all seeds evaluate the same test images."
                    )

            # Average each metric over 5 seeds per image_id
            avg_by_id: Dict[str, Dict[str, float]] = {}
            for img_id in ref_ids:
                metric_vals: Dict[str, List[float]] = {}
                for sr in seed_records:
                    for met, val in sr[img_id].items():
                        metric_vals.setdefault(met, []).append(float(val))
                avg_by_id[img_id] = {met: float(np.mean(vals))
                                     for met, vals in metric_vals.items()}
            per_loss[loss_key] = avg_by_id

        # R7: Validate identical image IDs across ALL losses
        first_key = next(iter(per_loss))
        ref_ids = sorted(per_loss[first_key].keys())
        for loss_key, by_id in per_loss.items():
            if sorted(by_id.keys()) != ref_ids:
                raise ValueError(
                    f"Image IDs differ between {first_key} and {loss_key}.\n"
                    f"Cannot perform paired analysis on misaligned images."
                )

        return per_loss

    # ── All 15 pairwise Wilcoxon tests (R2–R6) ───────────────────────────────

    def _run_pairwise_wilcoxon(self, avg_per_image: Dict) -> Dict:
        """
        All C(6,2)=15 pairwise comparisons per metric, Holm-corrected (R3, R4).
        STARE N<10 → descriptive only with explanation (R6).

        Returns
        -------
        Dict[metric → Dict["lossA__vs__lossB" → result_dict]]
        OR
        {"__skipped__": {reason, dataset, descriptive_only=True}}
        """
        try:
            from scipy import stats
        except ImportError:
            print("  [WARN] scipy not installed — skipping Wilcoxon tests.")
            print("         Install: pip install scipy")
            return {}

        loss_keys = list(LOSS_FUNCTIONS.keys())
        pairs     = list(itertools.combinations(loss_keys, 2))   # C(6,2) = 15
        image_ids = sorted(next(iter(avg_per_image.values())).keys())
        n_images  = len(image_ids)

        # R6: Skip inferential testing for small N (STARE N=4)
        if n_images < _MIN_N_FOR_WILCOXON:
            print(f"\n  ── Significance Testing — SKIPPED (R6) ──")
            print(f"  Dataset  : {self.cfg.name}  (N={n_images} test images)")
            print(f"  Reason   : N={n_images} < {_MIN_N_FOR_WILCOXON} — Wilcoxon signed-rank")
            print(f"             requires adequate sample size for valid inference.")
            print(f"             With only {n_images} pairs the test has extremely low")
            print(f"             statistical power (<0.20) and Type-I error control")
            print(f"             cannot be guaranteed. {self.cfg.name} results are")
            print(f"             reported DESCRIPTIVELY only (mean ± SD across seeds).")
            print(f"  Reference: Pratt (1959); Conover (1999) §4.2.")
            return {"__skipped__": {
                "reason": (f"N={n_images} test images < minimum {_MIN_N_FOR_WILCOXON} "
                           f"required for inferential testing"),
                "dataset": self.cfg.name,
                "descriptive_only": True,
            }}

        print(f"\n  ── Pairwise Wilcoxon Signed-Rank Tests ──")
        print(f"  Dataset      : {self.cfg.name}")
        print(f"  Unit         : per-image values averaged over {len(self.seeds)} seeds (R1)")
        print(f"  N images     : {n_images}")
        print(f"  Pairs tested : {len(pairs)}  [all C({len(loss_keys)},2) combinations]")
        print(f"  zero_method  : pratt (Pratt 1959)")
        print(f"  Effect size  : rank-biserial r_rb (Cureton 1956 / Kerby 2014) (R2)")
        print(f"  Correction   : Holm (1979) — PRIMARY (per metric family of {len(pairs)}) (R4)")
        print(f"  Supplementary: Benjamini-Hochberg FDR")
        print()

        all_results: Dict[str, Dict[str, Dict]] = {}

        for m in _ALL_METRICS:
            pair_results: Dict[str, Dict] = {}
            raw_pvals: List[float] = []
            pair_keys: List[str]   = []

            for (lossA, lossB) in pairs:
                pair_key = f"{lossA}__vs__{lossB}"

                # Pair strictly by image_id (R7)
                valid_ids = [
                    img_id for img_id in image_ids
                    if m in avg_per_image[lossA].get(img_id, {})
                    and m in avg_per_image[lossB].get(img_id, {})
                ]
                a_vals = np.array([avg_per_image[lossA][iid][m] for iid in valid_ids])
                b_vals = np.array([avg_per_image[lossB][iid][m] for iid in valid_ids])

                if len(a_vals) < 6:
                    pair_results[pair_key] = {
                        "skipped": True,
                        "reason":  f"n_valid_pairs={len(a_vals)} < 6",
                        "lossA": lossA, "lossB": lossB,
                    }
                    continue

                try:
                    res = stats.wilcoxon(a_vals, b_vals,
                                         alternative="two-sided",
                                         zero_method="pratt")
                    W = float(res.statistic)
                    p = float(res.pvalue)

                    # R2: Matched-pairs rank-biserial (Cureton 1956 / Kerby 2014)
                    r_rb = _matched_pairs_rank_biserial(a_vals, b_vals)

                    # Oriented differences: positive = lossA better than lossB
                    mean_diff   = float(np.mean(a_vals) - np.mean(b_vals))
                    median_diff = float(np.median(a_vals) - np.median(b_vals))
                    if m in _LOWER_IS_BETTER:
                        mean_diff   = -mean_diff
                        median_diff = -median_diff

                    pair_results[pair_key] = {
                        "lossA":               lossA,
                        "lossB":               lossB,
                        "W":                   round(W, 4),
                        "p_raw":               round(p, 8),
                        "rank_biserial":       round(r_rb, 6),
                        "mean_oriented_diff":  round(mean_diff, 6),
                        "median_oriented_diff": round(median_diff, 6),
                        "n_pairs":             len(a_vals),
                    }
                    raw_pvals.append(p)
                    pair_keys.append(pair_key)

                except Exception as exc:
                    pair_results[pair_key] = {
                        "lossA": lossA, "lossB": lossB,
                        "p_raw": None, "error": str(exc),
                    }

            # R4: Holm correction applied per metric family of 15 p-values
            if raw_pvals:
                p_holm = _holm_correction(raw_pvals)
                p_bh   = _bh_fdr(raw_pvals)
                for pk_idx, pk in enumerate(pair_keys):
                    pair_results[pk]["p_holm"]              = round(p_holm[pk_idx], 8)
                    pair_results[pk]["p_bh_supplementary"]  = round(p_bh[pk_idx], 8)
                    pair_results[pk]["significant_holm"]    = bool(p_holm[pk_idx] < 0.05)
                    pair_results[pk]["significant_bh"]      = bool(p_bh[pk_idx] < 0.05)

            all_results[m] = pair_results

        return all_results

    # ── Print summary ─────────────────────────────────────────────────────────

    def _print_summary_table(self, summary: Dict, pairwise: Dict) -> None:
        n_seeds = len(self.seeds)
        print(f"\n  ── Mean ± SD (N={n_seeds} seeds, ddof=1) ──")

        col_loss = 22
        col_val  = 18

        for section_metrics, section_label in [
            (_OVERLAP_METRICS, "Overlap metrics — higher is better"),
            (_TOPO_METRICS,    "Topology metrics — lower is better"),
        ]:
            print(f"\n  {section_label}:")
            print(f"  {'Loss Function':<{col_loss}}", end="")
            for m in section_metrics:
                print(f"  {_METRIC_LABELS[m]:<{col_val}}", end="")
            print()
            print("  " + "─" * (col_loss + (col_val + 2) * len(section_metrics)))
            for loss_key, per_metric in summary.items():
                label = LOSS_FUNCTIONS.get(loss_key, loss_key)
                print(f"  {label:<{col_loss}}", end="")
                if per_metric is None:
                    print("  (no data)")
                    continue
                for m in section_metrics:
                    info = per_metric.get(m)
                    if info is None:
                        print(f"  {'—':<{col_val}}", end="")
                    else:
                        cell = f"{info['mean']:.4f}±{info['std']:.4f}"
                        print(f"  {cell:<{col_val}}", end="")
                print()

        # Show Holm-significant pairs
        if not pairwise or "__skipped__" in pairwise:
            skip_info = pairwise.get("__skipped__", {})
            print(f"\n  Note: Significance testing skipped — {skip_info.get('reason', '')}")
            print(f"        {self.cfg.name} results are descriptive only.")
        else:
            print(f"\n  ── Holm-significant pairs (p_holm < 0.05) ──")
            found_any = False
            for m in _ALL_METRICS:
                mres = pairwise.get(m, {})
                sig_pairs = [(pk, v) for pk, v in mres.items()
                             if isinstance(v, dict) and v.get("significant_holm")]
                if sig_pairs:
                    found_any = True
                    print(f"\n  {_METRIC_LABELS.get(m, m)}:")
                    for pk, v in sig_pairs:
                        r  = v.get("rank_biserial", 0)
                        ph = v.get("p_holm", float("nan"))
                        print(f"    {pk:<44}  p_holm={ph:.4f}  r_rb={r:+.3f}")
            if not found_any:
                print("  No significant differences after Holm correction.")

    # ── Save outputs ──────────────────────────────────────────────────────────

    def _save_summary_json(self, summary: Dict) -> Path:
        out = {
            "dataset":  self.cfg.name,
            "seeds":    self.seeds,
            "n_seeds":  len(self.seeds),
            "note":     "mean and std computed from aggregate (per-seed) metrics; ddof=1",
            "summary":  summary,
        }
        path = self.out_dir / "multiseed_summary.json"
        with open(path, "w") as f:
            json.dump(out, f, indent=2)
        return path

    def _save_per_image_json(self, avg_per_image: Dict) -> Path:
        """Five-seed-averaged per-image values used as Wilcoxon input (R1)."""
        out = {
            "dataset":     self.cfg.name,
            "seeds":       self.seeds,
            "description": (
                "Per-image metrics averaged over all seeds (R1). "
                "These are the paired observations used in Wilcoxon tests."
            ),
            "data": avg_per_image,
        }
        path = self.out_dir / "per_image_averaged.json"
        with open(path, "w") as f:
            json.dump(out, f, indent=2)
        return path

    def _save_pairwise_json(self, pairwise: Dict) -> Path:
        out = {
            "dataset":      self.cfg.name,
            "seeds":        self.seeds,
            "test":         "Wilcoxon signed-rank, two-sided, zero_method=pratt",
            "correction_primary":       "Holm (1979) — per metric family of 15 pairs",
            "correction_supplementary": "Benjamini-Hochberg FDR (1995)",
            "effect_size":  "rank-biserial r_rb (Cureton 1956 / Kerby 2014)",
            "pairwise":     pairwise,
        }
        path = self.out_dir / "pairwise_wilcoxon.json"
        with open(path, "w") as f:
            json.dump(out, f, indent=2)
        return path

    def _save_pairwise_csv(self, pairwise: Dict) -> Path:
        """One row per metric × lossA × lossB — manuscript-ready."""
        header = ["metric", "lossA", "lossB", "pair_key",
                  "W", "p_raw", "p_holm", "significant_holm",
                  "p_bh_supplementary", "significant_bh",
                  "rank_biserial", "mean_oriented_diff", "median_oriented_diff",
                  "n_pairs"]
        rows = [header]

        if "__skipped__" in pairwise:
            info = pairwise["__skipped__"]
            rows.append(["SKIPPED", "", "", info.get("reason", ""),
                         "", "", "", "", "", "", "", "", "", ""])
        else:
            for m in _ALL_METRICS:
                mres = pairwise.get(m, {})
                for pair_key, v in mres.items():
                    if not isinstance(v, dict) or v.get("skipped"):
                        continue
                    lossA = v.get("lossA", "")
                    lossB = v.get("lossB", "")
                    W     = v.get("W", "")
                    rows.append([
                        m, lossA, lossB, pair_key,
                        f"{W:.4f}"  if isinstance(W, float) else "",
                        f"{v['p_raw']:.8f}"             if v.get("p_raw") is not None else "",
                        f"{v['p_holm']:.8f}"            if v.get("p_holm") is not None else "",
                        str(v.get("significant_holm", "")),
                        f"{v['p_bh_supplementary']:.8f}" if v.get("p_bh_supplementary") is not None else "",
                        str(v.get("significant_bh", "")),
                        f"{v['rank_biserial']:.6f}"     if v.get("rank_biserial") is not None else "",
                        f"{v['mean_oriented_diff']:.6f}" if v.get("mean_oriented_diff") is not None else "",
                        f"{v['median_oriented_diff']:.6f}" if v.get("median_oriented_diff") is not None else "",
                        str(v.get("n_pairs", "")),
                    ])

        path = self.out_dir / "pairwise_wilcoxon.csv"
        with open(path, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(",".join(str(x) for x in row) + "\n")
        return path

    # ── Box plots ─────────────────────────────────────────────────────────────

    def _plot_boxplots(self, summary: Dict) -> List[Path]:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print("  [WARN] matplotlib not installed — skipping boxplots.")
            return []

        paths: List[Path] = []
        colors = plt.cm.Set2(np.linspace(0, 1, len(summary)))

        for m in _ALL_METRICS:
            data_per_loss: List[List[float]] = []
            valid_labels:  List[str]         = []
            valid_colors:  List              = []
            for (loss_key, per_metric), color in zip(summary.items(), colors):
                if per_metric is None or m not in per_metric:
                    continue
                data_per_loss.append(per_metric[m]["raw"])
                valid_labels.append(LOSS_FUNCTIONS.get(loss_key, loss_key))
                valid_colors.append(color)

            if not data_per_loss:
                continue

            fig, ax = plt.subplots(figsize=(max(8, len(data_per_loss) * 1.5), 5))
            bp = ax.boxplot(data_per_loss, patch_artist=True, notch=False,
                            medianprops={"color": "black", "linewidth": 2})
            for patch, color in zip(bp["boxes"], valid_colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            for i, raw in enumerate(data_per_loss, 1):
                xs = np.random.normal(i, 0.05, size=len(raw))
                ax.scatter(xs, raw, s=30, color="navy", alpha=0.6, zorder=3)

            ax.set_xticks(range(1, len(valid_labels) + 1))
            ax.set_xticklabels(valid_labels, rotation=20, ha="right", fontsize=9)
            ax.set_ylabel(_METRIC_LABELS.get(m, m), fontsize=10)
            ax.set_title(
                f"Multi-Seed Distribution — {_METRIC_LABELS.get(m, m)}"
                f"\n{self.cfg.name}  |  N={len(self.seeds)} seeds",
                fontsize=11,
            )
            ax.grid(axis="y", alpha=0.3)
            fig.tight_layout()

            fname = self.out_dir / f"boxplot_{m}.png"
            fig.savefig(str(fname), dpi=300, bbox_inches="tight")
            plt.close(fig)
            paths.append(fname)
            print(f"  Boxplot: {fname.name}")

        return paths

    # ── Compact significance reports ──────────────────────────────────────────

    def _generate_compact_significance(self, pairwise: Dict) -> List[Path]:
        """Master method: WTL table, significance matrix PNG, significant pairs list.

        These three outputs present the same Wilcoxon results from pairwise_wilcoxon.json
        in formats suited for quick reading and manuscript inclusion.
        """
        paths: List[Path] = []

        # STARE N<10 → testing was skipped; write a brief note instead
        if "__skipped__" in pairwise:
            skip = pairwise["__skipped__"]
            p = self.out_dir / "significance_summary.txt"
            p.write_text(
                f"Statistical Significance Testing — {self.cfg.name}\n"
                f"{'='*60}\n\n"
                f"SKIPPED — insufficient sample size for valid inferential testing.\n"
                f"Reason : {skip.get('reason', '')}\n\n"
                f"Results for this dataset are DESCRIPTIVE ONLY.\n"
                f"Consult multiseed_summary.json for mean ± SD values.\n\n"
                f"Reference: Pratt (1959); Conover (1999) §4.2\n",
                encoding="utf-8",
            )
            paths.append(p)
            print(f"  [sig] Skipped note   : {p.name}")
            return paths

        p1, p2 = self._save_wtl_summary(pairwise)
        paths += [p1, p2]
        print(f"  [sig] WTL summary    : {p1.name}")
        print(f"  [sig] WTL CSV        : {p2.name}")

        p3 = self._save_significance_matrix_png(pairwise)
        if p3:
            paths.append(p3)
            print(f"  [sig] Matrix PNG     : {p3.name}")

        p4 = self._save_significant_pairs_txt(pairwise)
        paths.append(p4)
        print(f"  [sig] Sig. pairs TXT : {p4.name}")

        return paths

    # ── Win-Tie-Loss summary ──────────────────────────────────────────────────

    def _save_wtl_summary(self, pairwise: Dict):
        """Win-Tie-Loss table across all metrics.

        For each loss function, counts how many pairwise comparisons (across all
        metrics) it wins (significantly better), ties (not significant), or loses
        (significantly worse) after Holm correction.

        Returns (txt_path, csv_path).
        """
        loss_keys = list(LOSS_FUNCTIONS.keys())
        n_rivals  = len(loss_keys) - 1

        # wtl[loss_key] = [w, t, l]
        wtl: Dict[str, List[int]] = {k: [0, 0, 0] for k in loss_keys}

        for m in _ALL_METRICS:
            mres = pairwise.get(m, {})
            for v in mres.values():
                if not isinstance(v, dict) or v.get("skipped"):
                    continue
                lA  = v.get("lossA", "")
                lB  = v.get("lossB", "")
                if lA not in wtl or lB not in wtl:
                    continue
                sig = v.get("significant_holm", False)
                mod = v.get("mean_oriented_diff", 0.0)  # >0 → lossA better
                if not sig:
                    wtl[lA][1] += 1
                    wtl[lB][1] += 1
                elif mod > 0:
                    wtl[lA][0] += 1   # lossA wins
                    wtl[lB][2] += 1
                else:
                    wtl[lA][2] += 1
                    wtl[lB][0] += 1   # lossB wins

        # Sort rows by Net = W - L descending
        sorted_keys = sorted(loss_keys,
                             key=lambda k: wtl[k][0] - wtl[k][2],
                             reverse=True)

        total_per_loss = n_rivals * len(_ALL_METRICS)
        col_loss = 26

        hdr = (f"  {'Loss Function':<{col_loss}}  {'W':>5}  {'T':>5}  {'L':>5}  "
               f"{'Net':>5}  {'W%':>6}")
        sep = "  " + "─" * (col_loss + 33)

        data_rows = []
        for lk in sorted_keys:
            w, t, l = wtl[lk]
            net  = w - l
            wpct = 100.0 * w / total_per_loss if total_per_loss > 0 else 0.0
            data_rows.append(
                f"  {LOSS_FUNCTIONS.get(lk, lk):<{col_loss}}  {w:>5}  {t:>5}  "
                f"{l:>5}  {net:>+5}  {wpct:>5.1f}%"
            )

        legend = (
            "W = significantly better  |  T = not significant  |  "
            "L = significantly worse\n"
            f"Net = W − L  |  W% = W / (W+T+L)  |  "
            f"{total_per_loss} comparisons per loss "
            f"({n_rivals} rivals × {len(_ALL_METRICS)} metrics)\n"
            "Significance threshold: p_holm < 0.05  "
            "(Holm 1979 step-down correction, per metric family)"
        )

        txt = (
            f"Win-Tie-Loss Summary — {self.cfg.name}\n"
            f"Pairwise Wilcoxon signed-rank | Holm-corrected | seeds: {self.seeds}\n"
            f"{'='*72}\n\n"
            f"{hdr}\n{sep}\n"
            + "\n".join(data_rows)
            + f"\n\n{legend}\n"
        )

        p_txt = self.out_dir / "significance_wtl.txt"
        p_txt.write_text(txt, encoding="utf-8")

        # CSV: loss_key, label, W, T, L, Net, W_pct
        csv_rows = [["loss_key", "loss_label", "W", "T", "L", "Net", "W_pct"]]
        for lk in sorted_keys:
            w, t, l = wtl[lk]
            net  = w - l
            wpct = round(100.0 * w / total_per_loss, 1) if total_per_loss > 0 else 0.0
            csv_rows.append([lk, LOSS_FUNCTIONS.get(lk, lk), w, t, l, net, wpct])

        p_csv = self.out_dir / "significance_wtl.csv"
        p_csv.write_text(
            "\n".join(",".join(str(x) for x in r) for r in csv_rows),
            encoding="utf-8",
        )

        return p_txt, p_csv

    # ── Significance matrix PNG ───────────────────────────────────────────────

    def _cell_decision(self, mres: Dict, loss_keys: List[str], i: int, j: int):
        """Decision for matrix cell (row=i, col=j) relative to loss_keys ordering.

        Returns (decision, sig_marker_str, r_rb) where
        decision ∈ {"diag", "win", "loss", "tie", "skip"}.
        "win"  = loss_keys[i] significantly better than loss_keys[j]
        "loss" = loss_keys[i] significantly worse than loss_keys[j]
        """
        if i == j:
            return "diag", "", 0.0

        # Canonical pair order: itertools.combinations preserves list order,
        # so smaller index is always lossA in the stored pair key.
        if i < j:
            pair_key = f"{loss_keys[i]}__vs__{loss_keys[j]}"
            row_is_A = True
        else:
            pair_key = f"{loss_keys[j]}__vs__{loss_keys[i]}"
            row_is_A = False  # row is lossB in this stored pair

        v = mres.get(pair_key)
        if not isinstance(v, dict) or v.get("skipped"):
            return "skip", "", 0.0

        sig  = v.get("significant_holm", False)
        mod  = v.get("mean_oriented_diff", 0.0)  # >0 → lossA (in pair) better
        r_rb = v.get("rank_biserial", 0.0)
        p_h  = v.get("p_holm")

        if not sig:
            return "tie", "ns", r_rb

        a_wins   = mod > 0
        row_wins = a_wins if row_is_A else (not a_wins)
        return ("win" if row_wins else "loss"), _sig_marker(p_h), r_rb

    def _save_significance_matrix_png(self, pairwise: Dict) -> Optional[Path]:
        """Colour-coded 7×7 significance matrix for primary metrics (F1, AUC, Sensitivity, clDice).

        Green  = row loss significantly better than column loss (p_holm < 0.05).
        Red    = row loss significantly worse.
        Gray   = not significant.

        This is the most compact visual for manuscript inclusion.
        """
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import matplotlib.patches as mpatches
        except ImportError:
            print("  [WARN] matplotlib not installed — skipping significance matrix PNG.")
            return None

        loss_keys = list(LOSS_FUNCTIONS.keys())
        n = len(loss_keys)

        # Abbreviated axis labels
        short_labels = []
        for lk in loss_keys:
            lbl = LOSS_FUNCTIONS[lk]
            short_labels.append(lbl[:9] if " " not in lbl else lbl.split()[0][:9])

        _COLORS = {
            "win":  "#27ae60",   # green
            "loss": "#e74c3c",   # red
            "tie":  "#d5d8dc",   # light grey
            "diag": "#eaecee",   # diagonal
            "skip": "#fdfefe",   # white (missing data)
        }

        n_rows, n_cols = _VIZ_LAYOUT
        fig, axes = plt.subplots(n_rows, n_cols,
                                 figsize=(4.2 * n_cols, 5.2 * n_rows),
                                 squeeze=False)

        for panel_idx, metric in enumerate(_PRIMARY_VIZ_METRICS):
            row_idx = panel_idx // n_cols
            col_idx = panel_idx % n_cols
            ax   = axes[row_idx][col_idx]
            mres = pairwise.get(metric, {})

            for i in range(n):
                for j in range(n):
                    decision, marker, r_rb = self._cell_decision(mres, loss_keys, i, j)
                    fc   = _COLORS.get(decision, "#ffffff")
                    ypos = n - 1 - i   # row 0 at top

                    ax.add_patch(plt.Rectangle(
                        (j - 0.5, ypos - 0.5), 1, 1,
                        facecolor=fc, edgecolor="#aab7b8", linewidth=0.5,
                    ))

                    if decision == "diag":
                        ax.text(j, ypos, "—",
                                ha="center", va="center", fontsize=9, color="#7f8c8d")
                    elif decision in ("win", "loss"):
                        txt_col = "#fdfefe"
                        ax.text(j, ypos + 0.14, marker,
                                ha="center", va="center",
                                fontsize=9, fontweight="bold", color=txt_col)
                        ax.text(j, ypos - 0.20, f"r={r_rb:+.2f}",
                                ha="center", va="center", fontsize=6, color=txt_col)
                    elif decision == "tie":
                        ax.text(j, ypos, "ns",
                                ha="center", va="center", fontsize=8, color="#5d6d7e")
                    else:
                        ax.text(j, ypos, "n/a",
                                ha="center", va="center", fontsize=6.5, color="#aab7b8")

            ax.set_xlim(-0.5, n - 0.5)
            ax.set_ylim(-0.5, n - 0.5)
            ax.set_xticks(range(n))
            ax.set_yticks(range(n))
            ax.set_xticklabels(short_labels, rotation=40, ha="right", fontsize=7.5)
            ax.set_yticklabels(list(reversed(short_labels)), fontsize=7.5)
            # Mark lower-is-better metrics with ↓ in title
            title = _METRIC_LABELS.get(metric, metric)
            if metric in _LOWER_IS_BETTER:
                title += "  ↓"
            ax.set_title(title, fontsize=9.5, fontweight="bold", pad=8)
            if col_idx == 0:
                ax.set_ylabel("Loss A  (row)", fontsize=8)
            # Only show x-label on bottom row
            if row_idx == n_rows - 1:
                ax.set_xlabel("Loss B  (col)", fontsize=8)

        # Hide unused axes if n_metrics < n_rows * n_cols
        n_metrics = len(_PRIMARY_VIZ_METRICS)
        for extra in range(n_metrics, n_rows * n_cols):
            axes[extra // n_cols][extra % n_cols].set_visible(False)

        legend_handles = [
            mpatches.Patch(facecolor=_COLORS["win"],  edgecolor="#888",
                           label="Row sig. better (p_holm < 0.05)"),
            mpatches.Patch(facecolor=_COLORS["loss"], edgecolor="#888",
                           label="Row sig. worse (p_holm < 0.05)"),
            mpatches.Patch(facecolor=_COLORS["tie"],  edgecolor="#888",
                           label="Not significant (ns)"),
        ]
        fig.legend(handles=legend_handles, loc="lower center", ncol=3,
                   fontsize=9, bbox_to_anchor=(0.5, 0.0), framealpha=0.9)

        fig.suptitle(
            f"Pairwise Significance Matrix — {self.cfg.name}  "
            f"(Wilcoxon signed-rank, Holm-corrected, {len(self.seeds)} seeds)\n"
            f"Cell: * p<0.05  ** p<0.01  *** p<0.001  |  r = rank-biserial effect size  "
            f"|  ↓ = lower-is-better metric",
            fontsize=9, y=1.01,
        )

        plt.tight_layout(rect=[0, 0.05, 1, 1.0])

        out_path = self.out_dir / "significance_matrix.png"
        fig.savefig(str(out_path), dpi=200, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        return out_path

    # ── Significant pairs list ────────────────────────────────────────────────

    def _save_significant_pairs_txt(self, pairwise: Dict) -> Path:
        """Compact ranked list of only Holm-significant pairs.

        Sorted by metric order (_ALL_METRICS), then by |r_rb| descending so the
        strongest effects appear first within each metric group.
        """
        loss_keys     = list(LOSS_FUNCTIONS.keys())
        n_pairs_total = (len(loss_keys) * (len(loss_keys) - 1) // 2) * len(_ALL_METRICS)

        sig_rows = []
        for m in _ALL_METRICS:
            mres = pairwise.get(m, {})
            for pk, v in mres.items():
                if not isinstance(v, dict) or v.get("skipped"):
                    continue
                if not v.get("significant_holm", False):
                    continue
                mod  = v.get("mean_oriented_diff", 0.0)  # >0 → lossA better
                r_rb = v.get("rank_biserial", 0.0)
                sig_rows.append({
                    "metric":  m,
                    "winner":  v.get("lossA") if mod > 0 else v.get("lossB"),
                    "loser":   v.get("lossB") if mod > 0 else v.get("lossA"),
                    "W":       v.get("W", float("nan")),
                    "p_holm":  v.get("p_holm"),
                    "r_rb":    r_rb,
                    "effect":  _effect_label(r_rb),
                    "n_pairs": v.get("n_pairs", "?"),
                })

        metric_order = {m: i for i, m in enumerate(_ALL_METRICS)}
        sig_rows.sort(key=lambda r: (metric_order.get(r["metric"], 99), -abs(r["r_rb"])))

        CW = {"m": 18, "w": 24, "l": 24}  # column widths
        hdr = (
            f"  {'Metric':<{CW['m']}}  {'Winner':<{CW['w']}}  "
            f"{'Loser':<{CW['l']}}  {'W':>7}  {'p_holm':>9}  "
            f"{'r_rb':>6}  {'Effect':<12}  N"
        )
        sep = "  " + "─" * (CW["m"] + CW["w"] + CW["l"] + 44)

        body: List[str] = []
        if not sig_rows:
            body.append(
                "  No significant differences after Holm correction "
                "(all p_holm ≥ 0.05)."
            )
        else:
            body += [hdr, sep]
            prev_metric = None
            for r in sig_rows:
                if r["metric"] != prev_metric and prev_metric is not None:
                    body.append("")   # blank line between metric groups
                prev_metric = r["metric"]

                wlabel = LOSS_FUNCTIONS.get(r["winner"] or "", r["winner"] or "?")
                llabel = LOSS_FUNCTIONS.get(r["loser"]  or "", r["loser"]  or "?")
                W_val  = r["W"]
                W_str  = f"{W_val:.1f}" if W_val == W_val else "?"  # NaN → "?"
                p_str  = f"{r['p_holm']:.4f}" if r["p_holm"] is not None else "?"

                body.append(
                    f"  {_METRIC_LABELS.get(r['metric'], r['metric']):<{CW['m']}}  "
                    f"{wlabel[:CW['w']]:<{CW['w']}}  "
                    f"{llabel[:CW['l']]:<{CW['l']}}  "
                    f"{W_str:>7}  {p_str:>9}  "
                    f"{r['r_rb']:>+6.3f}  {r['effect']:<12}  {r['n_pairs']}"
                )

        footer = (
            f"\n  Found: {len(sig_rows)} significant pair(s) / {n_pairs_total} tested\n"
            f"\n  Effect size (|r_rb|): "
            f"Large ≥ 0.50 | Medium ≥ 0.30 | Small ≥ 0.10 | Negligible < 0.10\n"
            f"  Reference: Kerby (2014), Frontiers in Psychology.\n"
            f"  Direction: Winner = loss with higher oriented mean "
            f"(sign-corrected for lower-is-better metrics).\n"
        )

        txt = (
            f"Significant Pairs Summary — {self.cfg.name}\n"
            f"Wilcoxon signed-rank, two-sided | Holm-corrected (α = 0.05) | "
            f"seeds: {self.seeds}\n"
            f"Unit: per-image metrics averaged over {len(self.seeds)} seeds (R1)\n"
            f"{'='*90}\n\n"
            + "\n".join(body)
            + footer
        )

        out_path = self.out_dir / "significant_pairs.txt"
        out_path.write_text(txt, encoding="utf-8")
        return out_path
