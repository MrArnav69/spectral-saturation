"""Multi-statistic rerun.

Calls :mod:`src.statistics` against ``results/all_31_results.json`` and
writes one JSON containing every pooled ρ, p, N, K_sat, plus the
decoupling correlation, its bootstrap CI and the binary stopping-rule
AUC for every supported statistic.

Outputs
-------

* ``results/multistat_results.json`` — consolidated dump.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src import statistics as stats

results_dir = Path("results")
all_results_path = results_dir / "all_31_results.json"
out_path = results_dir / "multistat_results.json"

display_stats = ["S", "S_stable_rank", "S_two_nn", "S_mle"]
display_stat_labels = {
    "S":             "$S(\\mathrm{erank})$",
    "S_stable_rank": "$S(\\mathrm{stable\\_rank})$",
    "S_two_nn":      "$S(\\mathrm{TwoNN})$",
    "S_mle":         "$S(\\mathrm{MLE})$",
}


def main() -> None:
    if not all_results_path.exists():
        raise SystemExit(
            f"[multistat] abort: {all_results_path} missing — "
            "run `python analyses/pca_sweeps.py` first."
        )
    results_dir.mkdir(parents=True, exist_ok=True)
    all_results = stats.load_json(all_results_path.name)

    print(f"[multistat] Loaded {len(all_results)} tasks from {all_results_path}")

    summary: dict = {
        "n_tasks": len(all_results),
        "stats": {},
        "decoupling": stats.decoupling_corr(all_results),
        "decoupling_ci": {},
        "positive_count": {},
        "stopping_auc": {},
        "bootstrap_pooled_rho": {},
    }

    # ---- bootstrap CI on decoupling ρ --------------------------------
    erank_inf = []
    peaks = []
    for rows in all_results.values():
        if not rows:
            continue
        last = rows[-1]
        erank_inf.append(float(last.get("mean_erank", 0.0)))
        peaks.append(float(max(r["mean_acc"] for r in rows)))
    summary["decoupling_ci"] = stats.decoupling_ci(erank_inf, peaks)

    # ---- per-stat pooled ρ + positive-count + bootstrap CI -----------
    def _pos_count(stat: str) -> int:
        from src.statistics import per_task_spearman_multistat
        return sum(1 for r in all_results.values() if per_task_spearman_multistat(r, stat=stat)[0] > 0)

    def _median_per_task(stat: str) -> float:
        from src.statistics import per_task_spearman_multistat
        return float(np.median(
            [per_task_spearman_multistat(r, stat=stat)[0] for r in all_results.values()]
        ))

    for stat in display_stats:
        rho, p, n = stats.pooled_spearman_multistat(all_results, stat=stat)
        summary["stats"][stat] = {
            "pooled_rho": float(rho),
            "pooled_p":   float(p),
            "n_doubling_pairs": int(n),
            "positive_per_task": int(_pos_count(stat)),
            "median_rho": float(_median_per_task(stat)),
        }
        boot = stats.cluster_bootstrap_pooled_rho(
            all_results, stat=stat, B=10_000, seed=27,
        )
        summary["bootstrap_pooled_rho"][stat] = {
            "point": float(boot["point"]),
            "lo":    float(boot["lo"]),
            "hi":    float(boot["hi"]),
            "se":    float(boot["se"]),
        }
        # Stopping rule AUC — same definition across stats so the
        # direct head-to-head K_sat ranking is meaningful.
        summary["stopping_auc"][stat] = stats.binary_stopping_auc(
            all_results, stat=stat, threshold_geo_acc=1.25, signed=True,
        )
        summary["positive_count"][stat] = summary["stats"][stat]["positive_per_task"]

    summary["multistat_table"] = stats.multistat_summary_table(all_results).to_dict(orient="records")

    with open(out_path, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"[multistat] wrote {out_path}")
    for stat in display_stats:
        s = summary["stats"][stat]
        b = summary["bootstrap_pooled_rho"][stat]
        a = summary["stopping_auc"][stat]
        print(
            f"  {display_stat_labels[stat]:>22s}  "
            f"ρ={s['pooled_rho']:+.3f} (p={s['pooled_p']:.1e}, N={s['n_doubling_pairs']})  "
            f"95% CI [{b['lo']:+.3f}, {b['hi']:+.3f}]  "
            f"AUC={a['auc']:.3f}  pos={s['positive_per_task']}/{s['n_doubling_pairs']}"
        )


if __name__ == "__main__":
    main()
