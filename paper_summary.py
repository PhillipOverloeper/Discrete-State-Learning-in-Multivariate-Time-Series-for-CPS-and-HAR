
import ast
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

try:
    import numpy as np
except ImportError:
    np = None

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

# ============================================================
# CONFIGURATION
# ============================================================
INPUT_ROOT = "Daten - Backup/metrics"
OUTPUT_ROOT = "paper_results"

# Keep post-paper extensions such as RBM out of publication tables.
INCLUDE_DATASETS = []
INCLUDE_METHODS = ["vqvae", "catvae", "somvae", "hmm", "kmeans"]

# Main metrics you are likely to use in the paper
PRIMARY_METRICS = [
    "ari",
    "classification_accuracy",
    "classification_f1_macro",
    "forecasting_mse",
    "forecasting_r2_score",
    "nmi_sqrt",
    "purity",
    "v_measure",
]

# Metrics used to select the "best" number of states per (dataset, method)
SELECTION_METRICS = [
    "ari",
    "classification_accuracy",
    "forecasting_r2_score",
]

# Metrics where larger is better
HIGHER_IS_BETTER = {
    "ari",
    "classification_accuracy",
    "classification_f1_macro",
    "completeness",
    "homogeneity",
    "mutual_information",
    "nmi_min",
    "nmi_sqrt",
    "purity",
    "state_utilization",
    "top1_mass",
    "top5_mass",
    "top10_mass",
    "v_measure",
    "forecasting_r2_score",
}

# Metrics where smaller is better
LOWER_IS_BETTER = {
    "avg_label_entropy",
    "avg_state_entropy",
    "forecasting_mse",
    "test_kl",
    "unseen_states_in_test",
}

# Plot / formatting options
FIGSIZE_LINE = (7, 4.5)
FIGSIZE_HEATMAP = (7, 5)
DPI = 180
ROUND_DIGITS = 3
SAVE_PDF_PLOTS = True
MAKE_HEATMAPS = True
MAKE_LATEX = True

# ============================================================
# HELPERS
# ============================================================
def should_include(name, allow_list):
    return not allow_list or name in allow_list

def parse_num_states(folder_name):
    """
    Returns:
        num_states_raw: str
        num_states_total: int|None
        num_states_kind: str  # 'int' | 'tuple' | 'other'
    Examples:
        "4" -> ("4", 4, "int")
        "(2, 2)" -> ("(2, 2)", 4, "tuple")
    """
    raw = folder_name

    try:
        value = int(folder_name)
        return raw, value, "int"
    except Exception:
        pass

    try:
        parsed = ast.literal_eval(folder_name)
        if isinstance(parsed, tuple):
            total = 1
            for x in parsed:
                total *= int(x)
            return raw, total, "tuple"
    except Exception:
        pass

    return raw, None, "other"

def metric_direction(metric):
    if metric in HIGHER_IS_BETTER:
        return "higher"
    if metric in LOWER_IS_BETTER:
        return "lower"
    return "unknown"

def is_better(metric, a, b):
    """
    Returns True if a is better than b for the given metric.
    """
    direction = metric_direction(metric)
    if pd.isna(a):
        return False
    if pd.isna(b):
        return True
    if direction == "higher":
        return a > b
    if direction == "lower":
        return a < b
    return a > b

def format_mean_std(mean, std, digits=ROUND_DIGITS):
    if pd.isna(mean):
        return ""
    if pd.isna(std):
        return f"{mean:.{digits}f}"
    return f"{mean:.{digits}f} ± {std:.{digits}f}"

def safe_filename(text):
    return (
        str(text)
        .replace("/", "_")
        .replace("\\", "_")
        .replace(" ", "_")
        .replace("(", "")
        .replace(")", "")
        .replace(",", "-")
    )

def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)

def save_plot(fig, out_base):
    fig.tight_layout()
    fig.savefig(f"{out_base}.png", dpi=DPI, bbox_inches="tight")
    if SAVE_PDF_PLOTS:
        fig.savefig(f"{out_base}.pdf", bbox_inches="tight")
    plt.close(fig)

# ============================================================
# LOAD DATA
# ============================================================
def load_metrics(input_root):
    records = []

    input_path = Path(input_root)
    if not input_path.exists():
        raise FileNotFoundError(f"INPUT_ROOT does not exist: {input_root}")

    for dataset_dir in sorted(input_path.iterdir()):
        if not dataset_dir.is_dir():
            continue
        dataset = dataset_dir.name
        if not should_include(dataset, INCLUDE_DATASETS):
            continue

        for method_dir in sorted(dataset_dir.iterdir()):
            if not method_dir.is_dir():
                continue
            method = method_dir.name
            if not should_include(method, INCLUDE_METHODS):
                continue

            for states_dir in sorted(method_dir.iterdir()):
                if not states_dir.is_dir():
                    continue

                json_path = states_dir / "metrics_summary.json"
                if not json_path.exists():
                    continue

                num_states_raw, num_states_total, num_states_kind = parse_num_states(states_dir.name)

                with open(json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                for metric, values in data.items():
                    if not isinstance(values, dict):
                        continue

                    records.append(
                        {
                            "dataset": dataset,
                            "method": method,
                            "num_states_raw": num_states_raw,
                            "num_states_total": num_states_total,
                            "num_states_kind": num_states_kind,
                            "metric": metric,
                            "mean": values.get("mean"),
                            "std": values.get("std"),
                            "n": values.get("n"),
                            "source_file": str(json_path),
                        }
                    )

    df = pd.DataFrame(records)
    if df.empty:
        raise ValueError("No metrics_summary.json files found.")

    df = df.sort_values(
        ["dataset", "method", "metric", "num_states_total", "num_states_raw"],
        na_position="last",
    ).reset_index(drop=True)
    return df

# ============================================================
# EXPORT TABLES
# ============================================================
def export_long_and_wide(df, output_root):
    ensure_dir(output_root)

    long_csv = Path(output_root) / "metrics_long.csv"
    df.to_csv(long_csv, index=False)

    wide_means = df.pivot_table(
        index=["dataset", "method", "num_states_raw", "num_states_total"],
        columns="metric",
        values="mean",
        aggfunc="first",
    ).reset_index()

    wide_stds = df.pivot_table(
        index=["dataset", "method", "num_states_raw", "num_states_total"],
        columns="metric",
        values="std",
        aggfunc="first",
    ).reset_index()

    wide_means.to_csv(Path(output_root) / "metrics_wide_means.csv", index=False)
    wide_stds.to_csv(Path(output_root) / "metrics_wide_stds.csv", index=False)

    return wide_means, wide_stds

def export_metric_tables(df, output_root):
    tables_dir = Path(output_root) / "tables_per_metric"
    latex_dir = tables_dir / "latex"
    csv_dir = tables_dir / "csv"
    ensure_dir(tables_dir)
    ensure_dir(csv_dir)
    ensure_dir(latex_dir)

    for metric in sorted(df["metric"].unique()):
        df_m = df[df["metric"] == metric].copy()

        # Table with mean ± std strings
        df_m["mean_pm_std"] = df_m.apply(
            lambda row: format_mean_std(row["mean"], row["std"]),
            axis=1,
        )

        table = df_m.pivot_table(
            index=["dataset", "method"],
            columns="num_states_raw",
            values="mean_pm_std",
            aggfunc="first",
        )

        # Reorder columns using num_states_total when possible
        ordering = (
            df_m[["num_states_raw", "num_states_total"]]
            .drop_duplicates()
            .sort_values(["num_states_total", "num_states_raw"], na_position="last")
        )
        ordered_cols = [c for c in ordering["num_states_raw"].tolist() if c in table.columns]
        table = table.reindex(columns=ordered_cols)

        csv_path = csv_dir / f"{safe_filename(metric)}.csv"
        table.to_csv(csv_path)

        if MAKE_LATEX:
            tex_path = latex_dir / f"{safe_filename(metric)}.tex"
            latex_str = table.to_latex(escape=False, na_rep="")
            tex_path.write_text(latex_str, encoding="utf-8")

# ============================================================
# BEST STATE SELECTION
# ============================================================
def select_best_states(df, output_root):
    out_dir = Path(output_root) / "best_state_selection"
    ensure_dir(out_dir)

    summary_rows = []

    for selection_metric in SELECTION_METRICS:
        df_sel = df[df["metric"] == selection_metric].copy()
        if df_sel.empty:
            continue

        best_rows = []

        for (dataset, method), group in df_sel.groupby(["dataset", "method"]):
            group = group.copy()

            # Prefer rows with numeric num_states_total if available
            group = group.sort_values(
                ["num_states_total", "num_states_raw"],
                na_position="last"
            )

            best_idx = None
            best_val = None

            for idx, row in group.iterrows():
                current = row["mean"]
                if best_idx is None or is_better(selection_metric, current, best_val):
                    best_idx = idx
                    best_val = current

            if best_idx is not None:
                best = group.loc[best_idx]
                best_rows.append(
                    {
                        "dataset": dataset,
                        "method": method,
                        "selection_metric": selection_metric,
                        "best_num_states_raw": best["num_states_raw"],
                        "best_num_states_total": best["num_states_total"],
                        "best_score_mean": best["mean"],
                        "best_score_std": best["std"],
                    }
                )

        best_df = pd.DataFrame(best_rows)
        if not best_df.empty:
            best_df.to_csv(out_dir / f"best_states_by_{safe_filename(selection_metric)}.csv", index=False)
            summary_rows.append(best_df)

    if summary_rows:
        summary = pd.concat(summary_rows, ignore_index=True)
        summary.to_csv(out_dir / "all_best_state_selections.csv", index=False)

    # Build full metric tables at the selected num_states
    for selection_metric in SELECTION_METRICS:
        selection_file = out_dir / f"best_states_by_{safe_filename(selection_metric)}.csv"
        if not selection_file.exists():
            continue

        best_df = pd.read_csv(selection_file)
        merged = df.merge(
            best_df[["dataset", "method", "best_num_states_raw"]],
            left_on=["dataset", "method", "num_states_raw"],
            right_on=["dataset", "method", "best_num_states_raw"],
            how="inner",
        )

        if merged.empty:
            continue

        merged["mean_pm_std"] = merged.apply(
            lambda row: format_mean_std(row["mean"], row["std"]),
            axis=1,
        )

        pivot = merged.pivot_table(
            index=["dataset", "method"],
            columns="metric",
            values="mean_pm_std",
            aggfunc="first",
        ).reset_index()

        cols = ["dataset", "method"] + [m for m in PRIMARY_METRICS if m in pivot.columns] + [
            c for c in pivot.columns if c not in ["dataset", "method"] + PRIMARY_METRICS
        ]
        pivot = pivot[cols]
        pivot.to_csv(out_dir / f"primary_metrics_at_best_{safe_filename(selection_metric)}.csv", index=False)

        if MAKE_LATEX:
            latex = pivot.to_latex(index=False, escape=False, na_rep="")
            (out_dir / f"primary_metrics_at_best_{safe_filename(selection_metric)}.tex").write_text(
                latex, encoding="utf-8"
            )

# ============================================================
# PLOTS
# ============================================================
def plot_lineplots(df, output_root):
    out_dir = Path(output_root) / "plots" / "lineplots"
    ensure_dir(out_dir)

    for dataset in sorted(df["dataset"].unique()):
        df_dataset = df[df["dataset"] == dataset]

        for metric in sorted(df_dataset["metric"].unique()):
            df_m = df_dataset[df_dataset["metric"] == metric].copy()

            # For line plots we need numeric x values
            df_m = df_m[df_m["num_states_total"].notna()].copy()
            if df_m.empty:
                continue

            fig, ax = plt.subplots(figsize=FIGSIZE_LINE)

            for method in sorted(df_m["method"].unique()):
                sub = df_m[df_m["method"] == method].copy()
                sub = sub.sort_values(["num_states_total", "num_states_raw"])

                ax.plot(
                    sub["num_states_total"],
                    sub["mean"],
                    marker="o",
                    label=method,
                )

                if "std" in sub.columns:
                    y1 = sub["mean"] - sub["std"]
                    y2 = sub["mean"] + sub["std"]
                    ax.fill_between(sub["num_states_total"], y1, y2, alpha=0.15)

            ax.set_title(f"{dataset} — {metric}")
            ax.set_xlabel("Number of states")
            ax.set_ylabel(metric)
            ax.legend()
            ax.grid(True, alpha=0.25)

            out_base = out_dir / f"{safe_filename(dataset)}__{safe_filename(metric)}"
            save_plot(fig, out_base)

def plot_heatmaps(df, output_root):
    if not MAKE_HEATMAPS:
        return

    out_dir = Path(output_root) / "plots" / "heatmaps"
    ensure_dir(out_dir)

    for dataset in sorted(df["dataset"].unique()):
        df_dataset = df[df["dataset"] == dataset]

        for metric in sorted(df_dataset["metric"].unique()):
            df_m = df_dataset[df_dataset["metric"] == metric].copy()
            df_m = df_m[df_m["num_states_total"].notna()].copy()
            if df_m.empty:
                continue

            pivot = df_m.pivot_table(
                index="method",
                columns="num_states_total",
                values="mean",
                aggfunc="first",
            )

            if pivot.empty:
                continue

            fig, ax = plt.subplots(figsize=FIGSIZE_HEATMAP)

            if HAS_SEABORN:
                sns.heatmap(pivot, annot=True, fmt=f".{ROUND_DIGITS}f", ax=ax)
            else:
                im = ax.imshow(pivot.values, aspect="auto")
                ax.set_xticks(range(len(pivot.columns)))
                ax.set_xticklabels(pivot.columns)
                ax.set_yticks(range(len(pivot.index)))
                ax.set_yticklabels(pivot.index)
                for i in range(pivot.shape[0]):
                    for j in range(pivot.shape[1]):
                        val = pivot.iloc[i, j]
                        if pd.notna(val):
                            ax.text(j, i, f"{val:.{ROUND_DIGITS}f}", ha="center", va="center")
                fig.colorbar(im, ax=ax)

            ax.set_title(f"{dataset} — {metric}")
            ax.set_xlabel("Number of states")
            ax.set_ylabel("Method")

            out_base = out_dir / f"{safe_filename(dataset)}__{safe_filename(metric)}"
            save_plot(fig, out_base)

# ============================================================
# PAPER SUMMARY TABLES
# ============================================================
def export_overview_tables(df, output_root):
    out_dir = Path(output_root) / "overview_tables"
    ensure_dir(out_dir)

    # simple overview per dataset / method / num_states using primary metrics
    sub = df[df["metric"].isin(PRIMARY_METRICS)].copy()
    if sub.empty:
        return

    sub["mean_pm_std"] = sub.apply(
        lambda row: format_mean_std(row["mean"], row["std"]),
        axis=1,
    )

    overview = sub.pivot_table(
        index=["dataset", "method", "num_states_raw", "num_states_total"],
        columns="metric",
        values="mean_pm_std",
        aggfunc="first",
    ).reset_index()

    cols = ["dataset", "method", "num_states_raw", "num_states_total"] + [
        m for m in PRIMARY_METRICS if m in overview.columns
    ]
    overview = overview[cols]
    overview.to_csv(out_dir / "primary_metrics_overview.csv", index=False)

    if MAKE_LATEX:
        (out_dir / "primary_metrics_overview.tex").write_text(
            overview.to_latex(index=False, escape=False, na_rep=""),
            encoding="utf-8",
        )

# ============================================================
# MAIN
# ============================================================
def main():
    ensure_dir(OUTPUT_ROOT)

    print("Loading metrics...")
    df = load_metrics(INPUT_ROOT)

    print(f"Loaded {len(df)} metric rows.")
    print(f"Datasets: {sorted(df['dataset'].unique().tolist())}")
    print(f"Methods: {sorted(df['method'].unique().tolist())}")
    print(f"Metrics: {len(df['metric'].unique())} unique")

    print("Exporting long and wide CSV files...")
    export_long_and_wide(df, OUTPUT_ROOT)

    print("Exporting per-metric tables...")
    export_metric_tables(df, OUTPUT_ROOT)

    print("Selecting best number of states...")
    select_best_states(df, OUTPUT_ROOT)

    print("Creating overview tables...")
    export_overview_tables(df, OUTPUT_ROOT)

    print("Creating line plots...")
    plot_lineplots(df, OUTPUT_ROOT)

    print("Creating heatmaps...")
    plot_heatmaps(df, OUTPUT_ROOT)

    print(f"Done. Results written to: {OUTPUT_ROOT}")

if __name__ == "__main__":
    main()
