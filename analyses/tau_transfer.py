"""τ-transferability audit — leave-one-representation-out CV.

Loads the three representation-specific K-sweep JSONs as the LODO
folds, then runs :func:`src.tau_transfer.leave_one_rep_out_cv` to
determine whether the Youden-optimal threshold ``τ`` generalises across
representations (PCA, OpenCLIP ViT-B/32, OpenCLIP ViT-L/14).

Output
------

* ``results/tau_transfer_report.json`` — per-fold AUC plus the LODO
  summary table for whichever τ is in the sweep.

Run::

    python analyses/tau_transfer.py
"""

from __future__ import annotations

import json
from pathlib import Path

from src import tau_transfer
from src.protocols import load_results

results_dir = Path("results")
report_path = results_dir / "tau_transfer_report.json"

# One row per representation; ``all_31_results.json`` is the PCA fold,
# the two CLIP binary JSONs are the OpenCLIP folds.
representation_files: dict[str, str] = {
    "pca":      "all_31_results.json",
    "clip_b32": "clip_vitb32_binary_results.json",
    "clip_l14": "clip_vitl14_binary_results.json",
}


def _load_representations() -> dict[str, dict]:
    rep_results: dict[str, dict] = {}
    for rep_name, fname in representation_files.items():
        rows = load_results(results_dir / fname)
        if not rows:
            print(f"[tau_transfer]   {fname} missing or empty — "
                  f"rep '{rep_name}' will be empty.")
            rows = {}
        rep_results[rep_name] = rows
    return rep_results


def main() -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    rep_results = _load_representations()

    if not any(rep_results.values()):
        raise SystemExit(
            "[tau_transfer] abort: no representation results found — "
            "run `python analyses/pca_sweeps.py` and "
            "`python analyses/clip_sweeps.py` first."
        )

    df = tau_transfer.build_dataset(rep_results, stat="S")
    print(f"[tau_transfer] Built dataset with {len(df)} doubling-pair rows.")
    print(df.groupby("representation").size().rename("n_doubling_pairs"))

    cv_summary, raw_report = tau_transfer.leave_one_rep_out_cv(df, tau_sweep=None)
    tau_transfer.print_transfer_report(cv_summary)

    out = {
        "representations":   list(representation_files),
        "n_doubling_pairs":  int(len(df)),
        "lodo_summary":      cv_summary,
        "raw_report":        raw_report,
    }
    with open(report_path, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"[tau_transfer] wrote {report_path}")


if __name__ == "__main__":
    main()
