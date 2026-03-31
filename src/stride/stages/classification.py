# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "pandas",
#     "numpy",
#     "matplotlib",
#     "scikit-learn",
#     "scipy",
#     "pyyaml",
# ]
# ///
"""
Config-driven WT vs Tau logistic regression (Figure 7).

Approach (matching manuscript _shared.py):
  - Gait p-values: from GoPro batch M2 LMM (stride-level, speed+body corrected)
  - T-maze p-values: LMM where available, t-test as fallback (per-mouse level)
  - Features ranked by p-value, classifiers evaluated at K=1..N
  - Logistic regression with 5-fold stratified CV, balanced class weights

Usage:
    uv run python automated_pipeline/regression/run_regression.py
"""

import argparse
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


# ============================================================================
# FEATURE DEFINITIONS
# ============================================================================

TMAZE_FEATURES = [
    {"col": "percent_correct", "label": "Accuracy", "category": "T-maze"},
    {"col": "junction_explore_ms", "label": "Junction exploration", "category": "T-maze"},
    {"col": "num_probes", "label": "Snout probes", "category": "T-maze"},
    {"col": "mean_probe_dur_ms", "label": "Probe duration", "category": "T-maze"},
    {"col": "probe_bias_index", "label": "Probe bias", "category": "T-maze"},
    {"col": "probe_choice_corr", "label": "Probe-choice corr.", "category": "T-maze"},
    {"col": "choice_latency_log10", "label": "Choice latency", "category": "T-maze"},
    {"col": "stem_latency_log10", "label": "Stem latency", "category": "T-maze"},
]

GAIT_FEATURES = [
    {"col": "stride_length_euclid_cm", "label": "Stride length", "category": "Gait"},
    {"col": "step_freq_hz", "label": "Step frequency", "category": "Gait"},
    {"col": "nose_lat_disp_norm", "label": "Head sway", "category": "Gait"},
    {"col": "temporal_symmetry", "label": "Temporal symmetry", "category": "Gait"},
    {"col": "duration_ms", "label": "Stride duration", "category": "Gait"},
    {"col": "step_width_cm", "label": "Step width", "category": "Gait"},
]

ALL_FEATURES_DEF = {f["col"]: f for f in TMAZE_FEATURES + GAIT_FEATURES}


# ============================================================================
# DATA LOADING
# ============================================================================

def _standardize_mouse_id(df, id_cols, sep):
    if len(id_cols) == 1:
        df["MouseID"] = df[id_cols[0]].astype(str)
    else:
        df["MouseID"] = df[id_cols[0]].astype(str)
        for col in id_cols[1:]:
            df["MouseID"] = df["MouseID"] + sep + df[col].astype(str)
    return df


def _derive_tmaze_columns(df, fps):
    if "probes_L" in df.columns and "probes_R" in df.columns:
        df["num_probes"] = df["probes_L"] + df["probes_R"]
        total = df["num_probes"]
        df["probe_bias_index"] = np.where(
            total > 0, (df["probes_R"] - df["probes_L"]) / (total + 1e-12), np.nan)
    if "probe_frames_L" in df.columns and "probe_frames_R" in df.columns:
        total_frames = df["probe_frames_L"] + df["probe_frames_R"]
        total_probes = df.get("num_probes", 0)
        df["mean_probe_dur_ms"] = np.where(
            total_probes > 0, (total_frames / total_probes) / fps * 1000, np.nan)
    if "correct_bin" in df.columns:
        df["percent_correct"] = df["correct_bin"].astype(float)  # 0 or 1 at trial level
    if "latency_choice_ms" in df.columns:
        df["choice_latency_log10"] = np.log10(df["latency_choice_ms"].clip(lower=1))
    if "stem_latency_ms" in df.columns:
        df["stem_latency_log10"] = np.log10(df["stem_latency_ms"].clip(lower=1))
    return df


def load_per_mouse_features(config: dict) -> pd.DataFrame:
    """Load per-mouse feature matrix.

    If classification.merged_data_csv exists in config, load directly.
    Otherwise, build from raw stride + T-maze data.
    """
    class_cfg = config.get("classification", {})
    merged_path = class_cfg.get("merged_data_csv")

    if merged_path and Path(merged_path).exists():
        print(f"  Loading pre-built feature matrix: {merged_path}")
        data = pd.read_csv(merged_path)
        if "Label" not in data.columns:
            data["Label"] = (data["Genotype"] == "Tau").astype(int)
        # Ensure step_freq_hz exists
        if "step_freq_hz" not in data.columns and "duration_ms" in data.columns:
            data["step_freq_hz"] = 1000.0 / data["duration_ms"]
        print(f"  {len(data)} mice ({(data['Label']==0).sum()} WT, {(data['Label']==1).sum()} Tau)")
        return data

    # --- Build from raw data ---
    print("  Building feature matrix from raw data...")

    # Gait: each mouse from its own batch
    all_gait = []
    for batch_key, batch_cfg in config["batches"].items():
        stride_path = batch_cfg["stride_csv"]
        if not Path(stride_path).exists():
            continue
        print(f"  Gait ({batch_key}): {Path(stride_path).name}")
        gait = pd.read_csv(stride_path, low_memory=False)
        gait = _standardize_mouse_id(gait, batch_cfg["mouse_id_cols"],
                                      batch_cfg.get("mouse_id_sep", "_"))
        gait = gait[gait["Genotype"].isin(["WT", "Tau"])].dropna(subset=["Genotype"]).copy()

        # Speed filter: 15-25 cm/s (matching 01_exploratory_analysis.py)
        if "stride_speed_cm_s_mean" in gait.columns:
            before = len(gait)
            gait = gait[(gait["stride_speed_cm_s_mean"] >= 15) &
                        (gait["stride_speed_cm_s_mean"] <= 25)].copy()
            print(f"    Speed filter 15-25 cm/s: {before:,} -> {len(gait):,} strides")

        if "step_freq_hz" not in gait.columns:
            gait["step_freq_hz"] = 1000.0 / gait["duration_ms"]
        gait_cols = [f["col"] for f in GAIT_FEATURES if f["col"] in gait.columns]
        # Median aggregation (matching 01_exploratory_analysis.py)
        mouse = gait.groupby(["MouseID", "Genotype"])[gait_cols].median().reset_index()
        mouse["Batch"] = batch_cfg["name"]
        all_gait.append(mouse)
        print(f"    {mouse['MouseID'].nunique()} mice")
    gait_all = pd.concat(all_gait, ignore_index=True)

    # T-maze: pool all batches
    all_tmaze_trials = []
    for batch_key, batch_cfg in config["batches"].items():
        tmaze_dir = Path(batch_cfg.get("decisions_csv", "")).parent
        for candidate in [tmaze_dir / "learning_cycle_1.csv",
                          tmaze_dir / "metrics_meta.csv", tmaze_dir / "metrics.csv"]:
            if candidate.exists():
                df = pd.read_csv(candidate, low_memory=False)
                df = _standardize_mouse_id(df, batch_cfg["mouse_id_cols"],
                                            batch_cfg.get("mouse_id_sep", "_"))
                if "Genotype" in df.columns:
                    df = df[df["Genotype"].isin(["WT", "Tau"])].copy()
                df = _derive_tmaze_columns(df, batch_cfg.get("fps", 100))
                all_tmaze_trials.append(df)
                print(f"  T-maze ({batch_key}): {candidate.name} ({len(df)} trials)")
                break

    tmaze_all = pd.concat(all_tmaze_trials, ignore_index=True)
    # Per-mouse T-maze: percent_correct uses mean (proportion), rest use median
    tmaze_cols = [f["col"] for f in TMAZE_FEATURES if f["col"] in tmaze_all.columns]
    median_cols = [c for c in tmaze_cols if c != "percent_correct"]
    tmaze_mouse = tmaze_all.groupby(["MouseID"])[median_cols].median().reset_index()
    if "percent_correct" in tmaze_all.columns:
        pct = tmaze_all.groupby("MouseID")["percent_correct"].mean() * 100
        tmaze_mouse = tmaze_mouse.merge(pct.reset_index(), on="MouseID", how="left")

    # Probe-choice correlation
    if "probe_bias_index" in tmaze_all.columns and "choice_LR" in tmaze_all.columns:
        from scipy.stats import spearmanr
        tmaze_all["choice_numeric"] = tmaze_all["choice_LR"].map({"L": -1, "R": 1})
        pcc = []
        for mid in tmaze_all["MouseID"].unique():
            md = tmaze_all[tmaze_all["MouseID"] == mid].dropna(
                subset=["probe_bias_index", "choice_numeric"])
            if len(md) >= 5:
                rho, _ = spearmanr(md["probe_bias_index"], md["choice_numeric"])
                pcc.append({"MouseID": mid, "probe_choice_corr": rho})
        if pcc:
            tmaze_mouse = tmaze_mouse.merge(pd.DataFrame(pcc), on="MouseID", how="left")

    combined = gait_all.merge(tmaze_mouse, on="MouseID", how="left", suffixes=("", "_tm"))
    combined["Label"] = (combined["Genotype"] == "Tau").astype(int)
    print(f"\n  Feature matrix: {len(combined)} mice "
          f"({(combined['Label']==0).sum()} WT, {(combined['Label']==1).sum()} Tau)")
    return combined


# ============================================================================
# FEATURE RANKING (matching _shared.py approach)
# ============================================================================

def rank_features(data: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Rank features by p-value: LMM for gait, LMM/t-test for T-maze.

    Uses hardcoded p-values from config.yaml (authoritative LMM values),
    falls back to t-test for features without LMM p-values.
    """
    class_cfg = config.get("classification", {})

    # Load known LMM p-values from config
    gait_pvals = class_cfg.get("gait_lmm_pvalues", {})
    tmaze_pvals = class_cfg.get("tmaze_lmm_pvalues", {})
    known_pvals = {**gait_pvals, **tmaze_pvals}
    print(f"  LMM p-values from config: {len(known_pvals)} features")

    # Build feature stats
    rows = []
    all_features = [f["col"] for f in TMAZE_FEATURES + GAIT_FEATURES]
    available = [f for f in all_features if f in data.columns and data[f].notna().sum() > 5]

    for feat in available:
        info = ALL_FEATURES_DEF.get(feat, {"label": feat, "category": "Unknown"})
        wt = data[data["Genotype"] == "WT"][feat].dropna()
        tau = data[data["Genotype"] == "Tau"][feat].dropna()

        if feat in known_pvals:
            p = known_pvals[feat]
            source = "LMM"
        elif len(wt) >= 3 and len(tau) >= 3:
            _, p = stats.ttest_ind(wt, tau)
            source = "t-test"
        else:
            p = 1.0
            source = "N/A"

        diff = tau.mean() - wt.mean() if len(tau) > 0 and len(wt) > 0 else 0
        rows.append({
            "feature": feat,
            "label": info["label"],
            "category": info["category"],
            "p_value": p,
            "source": source,
            "direction": "Tau ↑" if diff > 0 else "Tau ↓",
            "significant": p < 0.05,
        })

    df = pd.DataFrame(rows).sort_values("p_value")
    return df


# ============================================================================
# CLASSIFICATION
# ============================================================================

def run_classification(data: pd.DataFrame, feature_df: pd.DataFrame, out_dir: Path,
                       config: dict):
    y = data["Label"].values
    sorted_features = feature_df["feature"].tolist()
    sig_features = feature_df[feature_df["significant"]]["feature"].tolist()

    print(f"\n  Features: {len(sorted_features)} total, {len(sig_features)} significant")
    for _, row in feature_df.iterrows():
        marker = "*" if row["significant"] else " "
        print(f"    {marker} {row['label']:25s} ({row['category']:6s}) "
              f"p={row['p_value']:.2e} [{row['source']}] {row['direction']}")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    def make_pipeline():
        """Logistic regression with scaling inside CV folds (no data leakage)."""
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")),
        ])

    def get_auc(X, y_):
        pipe = make_pipeline()
        scores = cross_val_score(pipe, X, y_, cv=cv, scoring="roc_auc")
        return scores.mean(), scores.std()

    def get_roc(X, y_):
        pipe = make_pipeline()
        y_probs = cross_val_predict(pipe, X, y_, cv=cv, method="predict_proba")[:, 1]
        fpr, tpr, _ = roc_curve(y_, y_probs)
        return fpr, tpr, auc(fpr, tpr)

    # --- AUC vs K ---
    print(f"\n  AUC vs K:")
    auc_results = []
    for k in range(1, len(sorted_features) + 1):
        X_k = data[sorted_features[:k]].values
        auc_mean, auc_std = get_auc(X_k, y)
        auc_results.append({"k": k, "auc_mean": auc_mean, "auc_std": auc_std,
                            "top_feature": sorted_features[k - 1]})
        print(f"    K={k:2d}: AUC={auc_mean:.3f}±{auc_std:.3f}  +{sorted_features[k-1]}")

    auc_df = pd.DataFrame(auc_results)
    auc_df.to_csv(out_dir / "regression_auc_vs_k.csv", index=False)

    best_k = int(auc_df.loc[auc_df["auc_mean"].idxmax(), "k"])
    optimal_features = sorted_features[:best_k]

    # --- Modality-specific AUCs ---
    sig_tmaze = [f for f in sig_features if ALL_FEATURES_DEF.get(f, {}).get("category") == "T-maze"]
    sig_gait = [f for f in sig_features if ALL_FEATURES_DEF.get(f, {}).get("category") == "Gait"]

    print(f"\n  Modality AUCs:")
    auc_tm = get_auc(data[sig_tmaze].values, y) if sig_tmaze else (0.5, 0)
    auc_ga = get_auc(data[sig_gait].values, y) if sig_gait else (0.5, 0)
    auc_opt = get_auc(data[optimal_features].values, y)
    auc_all = get_auc(data[sorted_features].values, y)

    print(f"    T-maze sig ({len(sig_tmaze)}): AUC={auc_tm[0]:.3f}")
    print(f"    Gait sig   ({len(sig_gait)}): AUC={auc_ga[0]:.3f}")
    print(f"    Optimal    (K={best_k}): AUC={auc_opt[0]:.3f}")
    print(f"    All feats  (K={len(sorted_features)}): AUC={auc_all[0]:.3f}")

    # --- ROC curves ---
    roc_data = {}
    if sig_tmaze:
        fpr, tpr, a = get_roc(data[sig_tmaze].values, y)
        roc_data["T-maze"] = (fpr, tpr, a, len(sig_tmaze))
    if sig_gait:
        fpr, tpr, a = get_roc(data[sig_gait].values, y)
        roc_data["Gait"] = (fpr, tpr, a, len(sig_gait))
    fpr, tpr, a = get_roc(data[sorted_features].values, y)
    roc_data["Combined"] = (fpr, tpr, a, len(sorted_features))

    # --- Figures ---
    colors = {"T-maze": "#3B82F6", "Gait": "#7B68AE", "Combined": "#10B981"}

    # AUC vs K
    fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
    ax.errorbar(auc_df["k"], auc_df["auc_mean"], yerr=auc_df["auc_std"],
                fmt="o-", color="#2C3E50", capsize=3, markersize=5)
    ax.axvline(best_k, color="#E74C3C", ls="--", alpha=0.7, label=f"Optimal K={best_k}")
    ax.set_xlabel("Number of features (K)", fontsize=11)
    ax.set_ylabel("ROC-AUC (5-fold CV)", fontsize=11)
    ax.set_title("WT vs Tau Classification: AUC vs Feature Count", fontsize=12, fontweight="bold")
    ax.set_ylim(0.4, 1.05)
    ax.legend()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for ext in ["png", "pdf"]:
        fig.savefig(out_dir / f"regression_auc_vs_k.{ext}", dpi=150, bbox_inches="tight")
    plt.close()

    # ROC curves
    fig, ax = plt.subplots(figsize=(5, 5), dpi=150)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="Chance")
    for domain, (fpr, tpr, a, k) in roc_data.items():
        ax.plot(fpr, tpr, color=colors.get(domain, "gray"), lw=2,
                label=f"{domain} (K={k}, AUC={a:.2f})")
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.set_title("WT vs Tau Classification", fontsize=12, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for ext in ["png", "pdf"]:
        fig.savefig(out_dir / f"regression_roc_curves.{ext}", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"\n  Saved figures to {out_dir}/")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="WT vs Tau logistic regression (Figure 7)")
    parser.add_argument("--config", default="automated_pipeline/config.yaml")
    parser.add_argument("--output", default="automated_pipeline/output")
    args = parser.parse_args()

    config = load_config(args.config)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Regression Pipeline — WT vs Tau Classification")
    print("=" * 60)

    data = load_per_mouse_features(config)
    feature_df = rank_features(data, config)
    feature_df.to_csv(out_dir / "regression_feature_ranking.csv", index=False)
    run_classification(data, feature_df, out_dir, config)

    print("\nDone.")


if __name__ == "__main__":
    main()
