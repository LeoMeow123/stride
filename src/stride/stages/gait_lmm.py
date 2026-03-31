# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "pandas",
#     "numpy",
#     "matplotlib",
#     "seaborn",
#     "statsmodels",
#     "pyyaml",
# ]
# ///
"""
Config-driven gait analysis pipeline (Stages 6-9).

Loads gait_per_stride_filtered.csv for each batch defined in config.yaml,
standardizes columns, tags trial phases, computes derived metrics,
runs M1/M2 LMMs, applies FDR correction, and outputs results + heatmaps.

Usage:
    uv run python automated_pipeline/gait/run_gait.py
    uv run python automated_pipeline/gait/run_gait.py --config automated_pipeline/config.yaml
    uv run python automated_pipeline/gait/run_gait.py --trial-phase task  # only task strides (default)
    uv run python automated_pipeline/gait/run_gait.py --trial-phase all   # all strides
"""

import argparse
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.formula.api as smf
import yaml
from matplotlib.colors import Normalize
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore")


# ============================================================================
# CONFIG LOADING
# ============================================================================

def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


# ============================================================================
# DATA LOADING & STANDARDIZATION
# ============================================================================

def load_batch(batch_key: str, batch_cfg: dict) -> pd.DataFrame:
    """Load and standardize a single batch's stride data."""
    path = batch_cfg["stride_csv"]
    # Fallback: if configured path doesn't exist, check pipeline output directory
    if not Path(path).exists():
        fallback = Path("automated_pipeline/output") / batch_key / "gait_per_stride_filtered.csv"
        if fallback.exists():
            path = str(fallback)
            print(f"  Using pipeline output: {path}")
    print(f"  Loading {batch_key}: {path}")
    df = pd.read_csv(path, low_memory=False)

    # Handle canonical column names from gait_extraction.py (mouse_id, sex, genotype)
    if "mouse_id" in df.columns and "MouseID" not in df.columns:
        df["MouseID"] = df["mouse_id"].astype(str)
    if "genotype" in df.columns and "Genotype" not in df.columns:
        df["Genotype"] = df["genotype"]
    if "sex" in df.columns and "Sex" not in df.columns:
        df["Sex"] = df["sex"]

    # Standardize MouseID (from batch-specific columns if not already set)
    if "MouseID" not in df.columns:
        id_cols = batch_cfg["mouse_id_cols"]
        sep = batch_cfg.get("mouse_id_sep", "_")
        if len(id_cols) == 1:
            df["MouseID"] = df[id_cols[0]].astype(str)
        else:
            df["MouseID"] = df[id_cols[0]].astype(str)
            for col in id_cols[1:]:
                df["MouseID"] = df["MouseID"] + sep + df[col].astype(str)

    # Standardize Genotype
    df = df.dropna(subset=["Genotype"]).copy()
    genotype_order = ["WT", "APP", "Tau"]
    df["Genotype"] = pd.Categorical(df["Genotype"], categories=genotype_order)

    # Standardize Sex if available
    if batch_cfg.get("has_sex", False) and "Sex" in df.columns:
        df["Sex"] = pd.Categorical(df["Sex"], categories=["F", "M"])

    # Add batch label
    df["batch"] = batch_cfg["name"]
    df["batch_key"] = batch_key

    n_strides = len(df)
    n_mice = df["MouseID"].nunique()
    print(f"    {n_strides:,} strides, {n_mice} mice")

    return df


def compute_derived_metrics(df: pd.DataFrame, metrics_cfg: list) -> pd.DataFrame:
    """Compute any derived metrics (e.g., step_freq_hz)."""
    for m in metrics_cfg:
        col = m["col"]
        if "derived" in m and col not in df.columns:
            expr = m["derived"]
            # Simple evaluation: "1000 / duration_ms"
            if "/" in expr:
                parts = expr.split("/")
                numerator = float(parts[0].strip())
                denominator_col = parts[1].strip()
                if denominator_col in df.columns:
                    df[col] = numerator / df[denominator_col]
                    print(f"    Computed {col} = {expr}")
        elif "derived" in m and col in df.columns:
            pass  # already exists
    return df


# ============================================================================
# TRIAL PHASE TAGGING (Stage 7)
# ============================================================================

def tag_trial_phases(df: pd.DataFrame, batch_cfg: dict) -> pd.DataFrame:
    """Tag each stride with trial_phase using events from T-maze behavioral analysis.

    Tags:
      - 'task': stride occurs between gate_frame and entry_frame
      - 'return': stride occurs outside valid trial window
      - 'poked_return': return stride with abnormally high speed
      - 'unmatched': stride could not be matched to any trial
    """
    decisions_path = batch_cfg.get("decisions_csv")
    if not decisions_path or not Path(decisions_path).exists():
        print(f"    No decisions.csv found — skipping trial phase tagging")
        df["trial_phase"] = "untagged"
        return df

    decisions = pd.read_csv(decisions_path, low_memory=False)
    print(f"    Loaded {len(decisions)} trials from {Path(decisions_path).name}")

    # Normalize video paths for matching (use filename stem)
    def normalize_path(p):
        if pd.isna(p):
            return ""
        return Path(str(p)).stem

    decisions["_video_key"] = decisions["video_path"].apply(normalize_path)
    df["_video_key"] = df["video_path"].apply(normalize_path) if "video_path" in df.columns else ""

    # Initialize
    df["trial_phase"] = "unmatched"

    # For each stride, check if it falls within a valid trial window
    for video_key in df["_video_key"].unique():
        if not video_key:
            continue
        stride_mask = df["_video_key"] == video_key
        trial_rows = decisions[decisions["_video_key"] == video_key]

        if trial_rows.empty:
            continue

        for _, trial in trial_rows.iterrows():
            gate = trial.get("gate_frame")
            entry = trial.get("entry_frame")
            if pd.isna(gate) or pd.isna(entry):
                continue

            # Strides within valid trial window
            task_mask = stride_mask & (df["frame_start"] >= gate) & (df["frame_start"] <= entry)
            df.loc[task_mask, "trial_phase"] = "task"

        # Everything not tagged as 'task' in this video is 'return'
        return_mask = stride_mask & (df["trial_phase"] == "unmatched")
        df.loc[return_mask, "trial_phase"] = "return"

    # Detect poked_return: return strides with speed > 2x mouse's task median
    for mouse_id in df["MouseID"].unique():
        mouse_mask = df["MouseID"] == mouse_id
        task_strides = df[mouse_mask & (df["trial_phase"] == "task")]
        return_strides = df[mouse_mask & (df["trial_phase"] == "return")]

        if task_strides.empty or return_strides.empty:
            continue
        if "stride_speed_cm_s_mean" not in df.columns:
            continue

        task_median_speed = task_strides["stride_speed_cm_s_mean"].median()
        if pd.isna(task_median_speed) or task_median_speed <= 0:
            continue

        speed_threshold = task_median_speed * 2.0
        poked_mask = (
            mouse_mask
            & (df["trial_phase"] == "return")
            & (df["stride_speed_cm_s_mean"] > speed_threshold)
        )
        df.loc[poked_mask, "trial_phase"] = "poked_return"

    # Cleanup temp column
    df.drop(columns=["_video_key"], inplace=True)

    # Report
    counts = df["trial_phase"].value_counts()
    for phase, count in counts.items():
        print(f"    {phase}: {count:,} strides ({100*count/len(df):.1f}%)")

    return df


# ============================================================================
# LMM ANALYSIS (Stage 9)
# ============================================================================

def run_lmm(data: pd.DataFrame, variable: str, model_name: str,
            model_cfg: dict, batch_cfg: dict, config: dict) -> dict | None:
    """Run a single LMM for one variable on one batch."""
    if variable not in data.columns:
        return None

    # Skip if outcome is a covariate in M2
    speed_col = config["covariates"]["speed"]
    if model_cfg.get("skip_outcome_if_covariate") and variable == speed_col:
        return None

    # Required columns
    covariates = model_cfg["covariates"]
    required = [variable, "Genotype", "MouseID"] + covariates
    if batch_cfg.get("has_sex", False) and "Sex" in data.columns:
        required.append("Sex")

    subset = data[required].dropna().copy()
    if len(subset) < 100:
        return None

    # Z-score outcome and covariates
    y_mean = subset[variable].mean()
    y_std = subset[variable].std()
    if y_std == 0:
        return None
    subset["y_z"] = (subset[variable] - y_mean) / y_std

    cov_terms = []
    for cov in covariates:
        z_col = f"{cov}_z"
        cov_mean = subset[cov].mean()
        cov_std = subset[cov].std()
        if cov_std == 0:
            continue
        subset[z_col] = (subset[cov] - cov_mean) / cov_std
        cov_terms.append(z_col)

    # Build formula
    formula_parts = ["y_z ~ Genotype"]
    if batch_cfg.get("has_sex", False) and "Sex" in subset.columns:
        formula_parts.append("Sex")
    formula_parts.extend(cov_terms)
    formula = " + ".join(formula_parts)

    ref_geno = config["statistics"]["reference_genotype"]
    optimizer = config["statistics"]["optimizer"]
    genotypes = [g for g in config["statistics"]["genotype_order"] if g != ref_geno]

    try:
        model = smf.mixedlm(formula, subset, groups=subset["MouseID"])
        result = model.fit(reml=True, method=optimizer)
        if result.bse.max() > 10:
            return None

        output = {
            "variable": variable,
            "model": model_name,
            "n_obs": len(subset),
            "n_mice": subset["MouseID"].nunique(),
            "converged": True,
        }

        for geno in genotypes:
            param = f"Genotype[T.{geno}]"
            if param in result.params.index:
                output[f"{geno}_coef"] = result.params[param] * y_std
                output[f"{geno}_se"] = result.bse[param] * y_std
                output[f"{geno}_pval"] = result.pvalues[param]
                output[f"{geno}_coef_z"] = result.params[param]  # standardized

        return output
    except Exception:
        return None


def run_all_lmms(data: pd.DataFrame, batch_key: str, batch_cfg: dict,
                 config: dict) -> pd.DataFrame:
    """Run M1 and M2 for all metrics on one batch."""
    metrics = [m["col"] for m in config["metrics"]]
    pretty = {m["col"]: m["label"] for m in config["metrics"]}
    results = []

    for model_name, model_cfg in config["models"].items():
        print(f"\n  Running {model_name} on {batch_cfg['name']}...")
        for variable in metrics:
            row = run_lmm(data, variable, model_name, model_cfg, batch_cfg, config)
            if row:
                row["batch"] = batch_cfg["name"]
                row["batch_key"] = batch_key
                row["label"] = pretty.get(variable, variable)
                results.append(row)

    if not results:
        return pd.DataFrame()

    df = pd.DataFrame(results)
    return df


# ============================================================================
# HEATMAP VISUALIZATION
# ============================================================================

ROW_ORDER = [
    "Tailtip Phase",
    "Tailbase Phase",
    "Snout Phase",
    "Tailtip Lateral Disp.",
    "Tailbase Lateral Disp.",
    "Snout Lateral Disp.",
    "Stride Length",
    "Step Width",
    "Step Length",
    "Stride Duration",
    "Step Frequency",
    "Temporal Symmetry",
    "Limb Duty Factor",
    "Angular Velocity",
]


def create_heatmap(results_df: pd.DataFrame, title: str, out_path: Path,
                   config: dict):
    """Create dual-panel heatmap: -log10(p) + effect size bubbles."""
    df = results_df[results_df["converged"] == True].copy()
    if len(df) == 0:
        return

    ref_geno = config["statistics"]["reference_genotype"]
    genotypes = [g for g in config["statistics"]["genotype_order"] if g != ref_geno]

    # Build matrices
    labels = df["label"].unique()
    Hc_pval = pd.DataFrame(index=labels, columns=genotypes, dtype=float)
    Bc = pd.DataFrame(index=labels, columns=genotypes, dtype=float)

    # Deduplicate: keep first occurrence of each label
    df = df.drop_duplicates(subset=["label"], keep="first")
    labels = df["label"].unique()
    Hc_pval = pd.DataFrame(index=labels, columns=genotypes, dtype=float)
    Bc = pd.DataFrame(index=labels, columns=genotypes, dtype=float)

    for _, row in df.iterrows():
        label = row["label"]
        for geno in genotypes:
            Hc_pval.at[label, geno] = row.get(f"{geno}_pval", np.nan)
            Bc.at[label, geno] = row.get(f"{geno}_coef_z", np.nan)

    Hc = -np.log10(Hc_pval.astype(float).clip(lower=np.finfo(float).tiny))

    # FDR correction
    all_pvals = Hc_pval.values.flatten()
    valid_mask = ~np.isnan(all_pvals)
    if valid_mask.any():
        _, fdr_vals, _, _ = multipletests(all_pvals[valid_mask], method="fdr_bh")
        all_fdr = np.full_like(all_pvals, np.nan)
        all_fdr[valid_mask] = fdr_vals
        Hc_FDR = pd.DataFrame(all_fdr.reshape(Hc_pval.shape),
                               index=Hc_pval.index, columns=Hc_pval.columns)
    else:
        Hc_FDR = Hc_pval.copy()

    # Reorder rows
    row_order = [r for r in ROW_ORDER if r in Hc.index]
    remaining = [r for r in Hc.index if r not in row_order]
    row_order = row_order + remaining
    Hc = Hc.reindex(row_order)
    Bc = Bc.reindex(row_order)
    Hc_pval = Hc_pval.reindex(row_order)
    Hc_FDR = Hc_FDR.reindex(row_order)

    nrows, ncols = len(Hc), len(Hc.columns)

    # --- Draw ---
    fig = plt.figure(figsize=(4.2, max(5, nrows * 0.38 + 1.2)), dpi=200)
    left_margin, panel_width, gap = 0.28, 0.26, 0.01
    bottom, height = 0.12, 0.75

    axL = fig.add_axes([left_margin, bottom, panel_width, height])
    axR = fig.add_axes([left_margin + panel_width + gap, bottom, panel_width, height])

    # LEFT: -log10(p)
    YELLOW = "#fff3b0"
    sns.heatmap(np.ones((nrows, ncols))[::-1], ax=axL, cmap=[YELLOW],
                vmin=0, vmax=1, cbar=False, linewidths=1.0, linecolor="white")

    pos = Hc.fillna(0).astype(float)
    vmax_pos = max(5.0, float(np.nanmax(pos.values)))
    sns.heatmap(pos.iloc[::-1].clip(lower=1.0), ax=axL,
                mask=(pos <= 0).iloc[::-1].values,
                cmap="YlGnBu", vmin=1.0, vmax=vmax_pos,
                linewidths=1.0, linecolor="white", cbar=False)

    for i, pheno in enumerate(Hc.index[::-1]):
        for j, col in enumerate(Hc.columns):
            p_val = float(Hc_pval.at[pheno, col])
            fdr_val = float(Hc_FDR.at[pheno, col])
            if not np.isfinite(p_val):
                continue
            stars = "***" if p_val < 1e-3 else ("**" if p_val < 1e-2 else ("*" if p_val < 5e-2 else ""))
            if stars:
                color = "black" if fdr_val < 0.05 else "#555555"
                axL.text(j + 0.5, i + 0.5, stars, ha="center", va="center",
                         color=color, fontsize=11, fontweight="bold")

    for sp in axL.spines.values():
        sp.set_visible(True); sp.set_linewidth(1.2); sp.set_color("black")
    axL.set_xticklabels(Hc.columns, rotation=0, fontsize=10)
    axL.set_yticklabels(Hc.index[::-1], rotation=0, fontsize=9)
    axL.set_title(u"\u2212log\u2081\u2080(p)", fontsize=10, fontweight="bold", pad=8)

    # RIGHT: Effect sizes
    sns.heatmap(np.zeros((nrows, ncols)), ax=axR, cmap="Greys",
                vmin=0, vmax=1, cbar=False, linewidths=1.0, linecolor="white")

    vals = Bc.values.astype(float)
    finite_vals = vals[np.isfinite(vals)]
    beta_lim = np.nanpercentile(np.abs(finite_vals), 95) if len(finite_vals) > 0 else 1.0
    beta_lim = max(beta_lim, 0.1)
    norm = Normalize(vmin=-beta_lim, vmax=beta_lim)
    cmap_effect = plt.get_cmap("RdBu_r")

    for i in range(nrows):
        for j in range(ncols):
            b = vals[nrows - 1 - i, j]
            if np.isfinite(b):
                size = 250 * (abs(b) / beta_lim) ** 0.85 + 35
                axR.scatter(j + 0.5, i + 0.5, s=size, color=cmap_effect(norm(b)),
                            alpha=0.9, edgecolors="black", linewidths=0.5)

    for sp in axR.spines.values():
        sp.set_visible(True); sp.set_linewidth(1.2); sp.set_color("black")
    axR.set_xticklabels(Bc.columns, rotation=0, fontsize=10)
    axR.set_yticklabels([]); axR.tick_params(axis="y", left=False)
    axR.set_title(u"Effect (\u03b2/SD)", fontsize=10, fontweight="bold", pad=8)

    # Colorbars
    cbar_left, cbar_width = 0.85, 0.015
    cax1 = fig.add_axes([cbar_left, 0.52, cbar_width, 0.35])
    sm1 = plt.cm.ScalarMappable(cmap="YlGnBu", norm=Normalize(vmin=1.0, vmax=vmax_pos))
    sm1.set_array([])
    plt.colorbar(sm1, cax=cax1).set_label(r"$-\log_{10}(p)$", fontsize=8)

    cax2 = fig.add_axes([cbar_left, 0.12, cbar_width, 0.35])
    sm2 = plt.cm.ScalarMappable(norm=norm, cmap=cmap_effect)
    sm2.set_array([])
    plt.colorbar(sm2, cax=cax2).set_label(u"\u03b2 (SD)", fontsize=8)

    n_obs = df["n_obs"].iloc[0]
    n_mice = df["n_mice"].iloc[0]
    fig.suptitle(f"{title}\nn = {n_obs:,} strides, {n_mice} mice",
                 fontsize=9, fontweight="bold", y=0.98)

    for ext in ["png", "pdf"]:
        fig.savefig(out_path.with_suffix(f".{ext}"), dpi=200, bbox_inches="tight")
    plt.close()
    print(f"    Saved: {out_path.name}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Config-driven gait LMM analysis")
    parser.add_argument("--config", default="automated_pipeline/config.yaml",
                        help="Path to config.yaml")
    parser.add_argument("--output", default="automated_pipeline/output",
                        help="Output directory")
    parser.add_argument("--trial-phase", default="task",
                        choices=["task", "all", "return", "poked_return"],
                        help="Which trial_phase to include (default: task)")
    args = parser.parse_args()

    config = load_config(args.config)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Gait Analysis Pipeline — Stages 7-9")
    print(f"Config: {args.config}")
    print(f"Output: {out_dir}")
    print(f"Trial phase filter: {args.trial_phase}")
    print("=" * 60)

    all_results = []

    for batch_key, batch_cfg in config["batches"].items():
        print(f"\n{'='*60}")
        print(f"Batch: {batch_cfg['name']}")
        print(f"{'='*60}")

        # Stage 6: Load filtered strides
        df = load_batch(batch_key, batch_cfg)

        # Stage 7: Trial phase tagging
        print("\n  Stage 7: Trial phase tagging...")
        df = tag_trial_phases(df, batch_cfg)

        # Filter by trial phase
        if args.trial_phase != "all":
            before = len(df)
            df = df[df["trial_phase"] == args.trial_phase].copy()
            print(f"  Filtered to trial_phase='{args.trial_phase}': {before:,} -> {len(df):,} strides")

        # Stage 8: Derived metrics
        print("\n  Stage 8: Derived metrics...")
        df = compute_derived_metrics(df, config["metrics"])

        # Stage 9: LMM
        print("\n  Stage 9: LMM analysis...")
        results = run_all_lmms(df, batch_key, batch_cfg, config)
        if len(results) > 0:
            all_results.append(results)

            # Generate heatmaps per model
            for model_name in config["models"]:
                model_results = results[results["model"] == model_name]
                if len(model_results) > 0:
                    title = f"{model_name} — {batch_cfg['name']}"
                    heatmap_path = out_dir / f"heatmap_{model_name}_{batch_key}"
                    create_heatmap(model_results, title, heatmap_path, config)

    # Save combined results
    if all_results:
        combined = pd.concat(all_results, ignore_index=True)
        csv_path = out_dir / "lmm_results_all.csv"
        combined.to_csv(csv_path, index=False)
        print(f"\nSaved combined results: {csv_path}")
        print(f"Total: {len(combined)} model fits across {combined['batch'].nunique()} batches")

        # Print significance summary
        print("\n" + "=" * 60)
        print("SIGNIFICANCE SUMMARY (p < 0.05)")
        print("=" * 60)
        ref_geno = config["statistics"]["reference_genotype"]
        genotypes = [g for g in config["statistics"]["genotype_order"] if g != ref_geno]
        for _, row in combined.iterrows():
            for geno in genotypes:
                p = row.get(f"{geno}_pval", 1.0)
                if pd.notna(p) and p < 0.05:
                    stars = "***" if p < 0.001 else ("**" if p < 0.01 else "*")
                    coef = row.get(f"{geno}_coef", 0)
                    print(f"  {row['batch']} | {row['model']} | {row['label']:25s} | "
                          f"{geno} vs {ref_geno}: coef={coef:+.4f}, p={p:.2e} {stars}")
    else:
        print("\nNo results generated.")

    print("\nDone.")


if __name__ == "__main__":
    main()
