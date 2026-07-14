from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# =========================
# Config
# =========================

plt.rcParams.update({
    "font.size": 16,
    "axes.titlesize": 20,
    "axes.labelsize": 18,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 16,
})
INPUT_CSV = Path("metrics_compiled_long.csv")
OUTPUT_FIG = Path("k_robustness_har_tep_ari_r2.pdf")

DATASETS = ["har", "tep"]
METRIC_MAP = {
    "metrics.ari": "ARI",
    "metrics.forecasting_r2_score": "R2",
    # In case your file uses a different name:
    "metrics.forecasting_r2_simulated": "R2",
}
METRICS_TO_PLOT = ["ARI", "R2"]

# Optional display names
DATASET_LABELS = {
    "har": "HAR",
    "tep": "TEP",
}
METHOD_LABELS = {
    "catvae": "CatVAE",
    "somvae": "SOM-VAE",
    "vqvae": "VQ-VAE",
    "kmeans": "k-means",
    "hmm": "HMM",
}

# For SOM-VAE like "(2,2)", "(3,3)", ...
def k_to_numeric(k: str) -> float:
    k = str(k).strip()
    if k.startswith("(") and k.endswith(")"):
        try:
            parts = [int(x.strip()) for x in k[1:-1].split(",")]
            prod = 1
            for p in parts:
                prod *= p
            return float(prod)
        except Exception:
            return float("nan")
    try:
        return float(k)
    except Exception:
        return float("nan")


def main() -> None:
    df = pd.read_csv(INPUT_CSV, dtype={"K": str})

    df = df[df["method"].str.lower() != "node"].copy()
    # Normalize fields
    df["dataset"] = df["dataset"].astype(str).str.lower().str.strip()
    df["method"] = df["method"].astype(str).str.lower().str.strip()
    df["metric"] = df["metric"].astype(str).str.strip()
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df["K"] = df["K"].astype(str).str.strip()

    # Keep only desired metrics
    df = df[df["metric"].isin(METRIC_MAP)].copy()
    df["metric_clean"] = df["metric"].map(METRIC_MAP)

    # Keep only selected datasets and metrics
    df = df[df["dataset"].isin(DATASETS) & df["metric_clean"].isin(METRICS_TO_PLOT)].copy()

    # Convert K to numeric plotting axis
    df["K_numeric"] = df["K"].map(k_to_numeric)
    df = df[df["K_numeric"].notna()].copy()

    # Aggregate across seeds
    summary = (
        df.groupby(["dataset", "metric_clean", "method", "K", "K_numeric"], as_index=False)
          .agg(
              mean=("value", "mean"),
              std=("value", "std"),
              n=("value", "count"),
          )
          .sort_values(["dataset", "metric_clean", "method", "K_numeric"])
    )

    # Create 2x2 figure
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=False)
    axes = axes.flatten()

    panel_order = [
        ("har", "ARI"),
        ("har", "R2"),
        ("tep", "ARI"),
        ("tep", "R2"),
    ]

    for ax, (dataset, metric) in zip(axes, panel_order):
        sub = summary[(summary["dataset"] == dataset) & (summary["metric_clean"] == metric)].copy()

        for method, g in sub.groupby("method"):
            g = g.sort_values("K_numeric")
            label = METHOD_LABELS.get(method, method)
            ax.plot(g["K_numeric"], g["mean"], marker="o", label=label)
            ax.fill_between(
                g["K_numeric"],
                g["mean"] - g["std"].fillna(0),
                g["mean"] + g["std"].fillna(0),
                alpha=0.15,
            )

        ax.set_title(f"{DATASET_LABELS.get(dataset, dataset)} – {metric}")
        ax.set_xlabel("K")
        ax.set_ylabel(metric)
        ax.grid(True, alpha=0.3)

    # Put legend once
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5, frameon=False)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(OUTPUT_FIG, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved figure to: {OUTPUT_FIG.resolve()}")


if True:
    main()