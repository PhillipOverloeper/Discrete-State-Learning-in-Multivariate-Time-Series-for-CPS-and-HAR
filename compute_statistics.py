from __future__ import annotations

import warnings
from pathlib import Path

import pandas as pd
from scipy.stats import wilcoxon

# =========================
# Configuration
# =========================
INPUT_CSV = Path("metrics_compiled_long.csv")
OUTPUT_DIR = Path("analysis_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

# Dataset-specific fixed K values used in the main paper
FIXED_K = {
    "har": "6",
    "pamap2": "12",
    "mhealth": "12",
    "tep": "20",
    "wadi": "9",
}

# The RBM was added after the ETFA paper and must not enter paper statistics.
PAPER_METHODS = {"vqvae", "catvae", "somvae", "hmm", "kmeans"}

# Map raw metric names from your CSV to cleaner paper names
METRIC_MAP = {
    "metrics.ari": "ARI",
    "metrics.v_measure": "V",
    "metrics.classification_f1_macro": "F1",
    "metrics.classification_accuracy": "Accuracy",
    "metrics.forecasting_r2_score": "R2",
    "metrics.forecasting_mse": "MSE",
    "metrics.anomaly_auc_roc": "AUROC",
}

# Metrics where higher values are better
HIGHER_IS_BETTER = {"ARI", "V", "F1", "Accuracy", "R2", "AUROC"}

# Set to True if you want to keep only the mapped metrics
DROP_UNMAPPED_METRICS = True


# =========================
# Helpers
# =========================
def load_and_prepare(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, dtype={"K": str})

    required_cols = {"dataset", "task", "metric", "K", "method", "seed", "value"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Input CSV is missing required columns: {sorted(missing)}")

    df["dataset"] = df["dataset"].astype(str).str.strip().str.lower()
    df["task"] = df["task"].astype(str).str.strip()
    df["metric"] = df["metric"].astype(str).str.strip()
    df["K"] = df["K"].astype(str).str.strip()
    df["method"] = df["method"].astype(str).str.strip().str.lower()
    df["seed"] = pd.to_numeric(df["seed"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    df = df.dropna(subset=["seed", "value"]).copy()
    df = df[df["method"].isin(PAPER_METHODS)].copy()
    df["seed"] = df["seed"].astype(int)

    df["metric_clean"] = df["metric"].map(METRIC_MAP)

    if DROP_UNMAPPED_METRICS:
        df = df[df["metric_clean"].notna()].copy()
    else:
        df["metric_clean"] = df["metric_clean"].fillna(df["metric"])

    return df


def summarize(df: pd.DataFrame, out_path: Path) -> pd.DataFrame:
    summary = (
        df.groupby(["dataset", "task", "metric_clean", "K", "method"], as_index=False)
          .agg(
              mean=("value", "mean"),
              std=("value", "std"),
              n=("value", "count"),
          )
          .sort_values(["dataset", "task", "metric_clean", "K", "mean"], ascending=[True, True, True, True, False])
    )
    summary.to_csv(out_path, index=False)
    return summary


def filter_fixed_k(df: pd.DataFrame, fixed_k: dict[str, str]) -> pd.DataFrame:
    return df[df.apply(lambda r: fixed_k.get(r["dataset"]) == str(r["K"]), axis=1)].copy()


def run_top2_wilcoxon(df_fixed: pd.DataFrame, out_path: Path) -> pd.DataFrame:
    """
    For each dataset x metric_clean, find the top 2 methods by mean score
    and run a paired Wilcoxon signed-rank test across matched seeds.
    """
    means = (
        df_fixed.groupby(["dataset", "metric_clean", "method"], as_index=False)["value"]
               .mean()
    )

    rows = []

    for (dataset, metric), grp in means.groupby(["dataset", "metric_clean"]):
        ascending = metric not in HIGHER_IS_BETTER
        grp = grp.sort_values("value", ascending=ascending).reset_index(drop=True)

        if len(grp) < 2:
            continue

        best_method = grp.iloc[0]["method"]
        second_method = grp.iloc[1]["method"]

        a = df_fixed[
            (df_fixed["dataset"] == dataset) &
            (df_fixed["metric_clean"] == metric) &
            (df_fixed["method"] == best_method)
        ][["seed", "value"]].rename(columns={"value": "value_best"})

        b = df_fixed[
            (df_fixed["dataset"] == dataset) &
            (df_fixed["metric_clean"] == metric) &
            (df_fixed["method"] == second_method)
        ][["seed", "value"]].rename(columns={"value": "value_second"})

        merged = a.merge(b, on="seed", how="inner").sort_values("seed")
        n_pairs = len(merged)

        if n_pairs < 2:
            rows.append({
                "dataset": dataset,
                "metric": metric,
                "best_method": best_method,
                "second_method": second_method,
                "best_mean": a["value_best"].mean() if "value_best" in a else None,
                "second_mean": b["value_second"].mean() if "value_second" in b else None,
                "mean_diff": None,
                "n_pairs": n_pairs,
                "wilcoxon_stat": None,
                "p_value": None,
                "note": "not enough paired seeds",
            })
            continue

        diffs = merged["value_best"] - merged["value_second"]
        mean_diff = diffs.mean()

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                stat, p_value = wilcoxon(merged["value_best"], merged["value_second"], alternative="two-sided")
            note = ""
        except ValueError as e:
            stat, p_value = None, 1.0
            note = f"wilcoxon_failed: {e}"

        rows.append({
            "dataset": dataset,
            "metric": metric,
            "best_method": best_method,
            "second_method": second_method,
            "best_mean": merged["value_best"].mean(),
            "second_mean": merged["value_second"].mean(),
            "mean_diff": mean_diff,
            "n_pairs": n_pairs,
            "wilcoxon_stat": stat,
            "p_value": p_value,
            "note": note,
        })

    result = pd.DataFrame(rows)

    pvals = result["p_value"].tolist()
    valid = [(i, p) for i, p in enumerate(pvals) if pd.notna(p)]
    m = len(valid)

    sorted_valid = sorted(valid, key=lambda x: x[1])
    holm_adj = [None] * len(pvals)

    prev = 0.0
    for rank, (idx, p) in enumerate(sorted_valid, start=1):
        adj = (m - rank + 1) * p
        adj = max(adj, prev)
        adj = min(adj, 1.0)
        holm_adj[idx] = adj
        prev = adj

    result["p_value_holm"] = holm_adj
    result["significant_holm"] = result["p_value_holm"].apply(lambda x: x < 0.05 if pd.notna(x) else None)

    result.to_csv(out_path, index=False)
    return result


def compute_average_rank(df: pd.DataFrame, out_path: Path) -> pd.DataFrame:
    """
    Average method rank across all K values for each dataset x metric_clean.
    """
    means = (
        df.groupby(["dataset", "metric_clean", "K", "method"], as_index=False)["value"]
          .mean()
    )

    means["rank"] = means.groupby(["dataset", "metric_clean", "K"]).apply(
        lambda g: g["value"].rank(
            ascending=(g.name[1] not in HIGHER_IS_BETTER),
            method="average"
        )
    ).reset_index(level=[0, 1, 2], drop=True)

    rank_summary = (
        means.groupby(["dataset", "metric_clean", "method"], as_index=False)
             .agg(
                 avg_rank=("rank", "mean"),
                 avg_value=("value", "mean"),
                 num_K=("K", "nunique"),
             )
             .sort_values(["dataset", "metric_clean", "avg_rank", "avg_value"], ascending=[True, True, True, False])
    )

    rank_summary.to_csv(out_path, index=False)
    return rank_summary


def compute_win_counts(df: pd.DataFrame, out_path: Path) -> pd.DataFrame:
    """
    Count how often each method wins across K values for each dataset x metric_clean.
    """
    means = (
        df.groupby(["dataset", "metric_clean", "K", "method"], as_index=False)["value"]
          .mean()
    )

    winners = []
    for (dataset, metric, k), grp in means.groupby(["dataset", "metric_clean", "K"]):
        ascending = metric not in HIGHER_IS_BETTER
        grp = grp.sort_values("value", ascending=ascending).reset_index(drop=True)
        best = grp.iloc[0]
        winners.append({
            "dataset": dataset,
            "metric_clean": metric,
            "K": k,
            "winner_method": best["method"],
            "winner_value": best["value"],
        })

    winners_df = pd.DataFrame(winners)

    win_counts = (
        winners_df.groupby(["dataset", "metric_clean", "winner_method"], as_index=False)
                  .size()
                  .rename(columns={"winner_method": "method", "size": "num_wins"})
                  .sort_values(["dataset", "metric_clean", "num_wins"], ascending=[True, True, False])
    )

    win_counts.to_csv(out_path, index=False)
    return win_counts


def build_paper_friendly_table(summary_fixed: pd.DataFrame, out_path: Path) -> pd.DataFrame:
    """
    Build a compact CSV with formatted 'mean ± std' strings.
    """
    tbl = summary_fixed.copy()
    tbl["formatted"] = tbl.apply(
        lambda r: f"{r['mean']:.3f} ± {r['std']:.3f}" if pd.notna(r["std"]) else f"{r['mean']:.3f}",
        axis=1
    )
    tbl.to_csv(out_path, index=False)
    return tbl


def main() -> None:
    print(f"Reading: {INPUT_CSV.resolve()}")
    df = load_and_prepare(INPUT_CSV)

    print("\nDetected cleaned metrics:")
    for m in sorted(df["metric_clean"].dropna().unique()):
        print(f"  - {m}")

    # 1) All results summary
    summarize(df, OUTPUT_DIR / "metrics_summary_all.csv")
    print(f"\nSaved: {OUTPUT_DIR / 'metrics_summary_all.csv'}")

    # 2) Fixed-K summary
    df_fixed = filter_fixed_k(df, FIXED_K)
    summary_fixed = summarize(df_fixed, OUTPUT_DIR / "metrics_summary_fixedK.csv")
    build_paper_friendly_table(summary_fixed, OUTPUT_DIR / "metrics_summary_fixedK_formatted.csv")
    print(f"Saved: {OUTPUT_DIR / 'metrics_summary_fixedK.csv'}")
    print(f"Saved: {OUTPUT_DIR / 'metrics_summary_fixedK_formatted.csv'}")

    # 3) Significance tests at fixed K
    top2_tests = run_top2_wilcoxon(df_fixed, OUTPUT_DIR / "significance_top2_fixedK.csv")
    print(f"Saved: {OUTPUT_DIR / 'significance_top2_fixedK.csv'}")

    # 4) K-sensitivity: average rank
    avg_rank = compute_average_rank(df, OUTPUT_DIR / "k_sensitivity_average_rank.csv")
    print(f"Saved: {OUTPUT_DIR / 'k_sensitivity_average_rank.csv'}")

    # 5) K-sensitivity: win counts
    win_counts = compute_win_counts(df, OUTPUT_DIR / "k_sensitivity_win_counts.csv")
    print(f"Saved: {OUTPUT_DIR / 'k_sensitivity_win_counts.csv'}")

    # Short console preview
    print("\n=== Fixed-K summary preview ===")
    print(summary_fixed.head(20).to_string(index=False))

    print("\n=== Significance preview ===")
    if top2_tests.empty:
        print("No significance results generated.")
    else:
        print(top2_tests.head(20).to_string(index=False))

    print("\n=== K-sensitivity average-rank preview ===")
    print(avg_rank.head(20).to_string(index=False))

    print("\n=== K-sensitivity win-count preview ===")
    print(win_counts.head(20).to_string(index=False))


if True:
    main()
