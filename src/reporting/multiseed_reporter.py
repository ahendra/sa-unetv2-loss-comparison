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
  3. pairwise_wilcoxon.json     — all 15 pairs per metric, Holm-corrected
  4. pairwise_wilcoxon.csv      — manuscript-friendly CSV

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


def _sig_marker(p: Optional[float]) -> str:
    if p is None:
        return "n/a"
    for threshold, marker in _SIG_MARKERS:
        if p < threshold:
            return marker
    return "ns"


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
