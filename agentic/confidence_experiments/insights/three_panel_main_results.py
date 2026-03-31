"""Three-panel figure: ridgeplot, scatterplot, and sample-size agreement."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from adjustText import adjust_text
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch
from scipy.stats import gaussian_kde

from checks import bootstrap_mean_test, overlap_coefficient

# ---------------------------------------------------------------------------
# Style & constants
# ---------------------------------------------------------------------------
sns.set_style("whitegrid")

NULL_COLOR = "0.6"
ALT_COLOR = "#FF7F50"

# Display names (match LaTeX \newcommand names)
siglow = "Passed Both Checks"
nonsiglow = "Failed Yes Check"
sighigh = "Failed Overlap Check"
nonsighigh = "Failed Both Checks"

CAT_COLORS = {
    siglow: "seagreen",
    nonsiglow: "goldenrod",
    sighigh: "mediumpurple",
    nonsighigh: "lightcoral",
}
CATEGORIES = list(CAT_COLORS)

# Internal classification -> display name
CAT_DISPLAY = {
    "Confident Yes": siglow,
    "Low Signal": nonsiglow,
    "Analysis Failure": sighigh,
    "No Signal": nonsighigh,
}

OVL_THRESHOLD = 0.2
MEAN_ALPHA = 0.05

X_GRID = np.linspace(0, 100, 500)
ROW_HEIGHT = 1.0
RIDGE_SCALE = 0.75

# Font sizes (unified across panels)
LABEL_FS = 14
TICK_FS = 11
LEGEND_FS = 10
ANNOT_FS = 11
TITLE_FS = 14

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
BASE_DIR = Path("..").resolve()
CSV_PATH = BASE_DIR / "aggregated_results" / "aggregated_results.csv"
FIG_DIR = Path("figures")
FIG_DIR.mkdir(exist_ok=True)

df = pd.read_csv(CSV_PATH, keep_default_na=False)
df_perturbed = df[df["perturbation"] != "none"].copy()
datasets = sorted(df_perturbed["dataset"].unique())
n_ds = len(datasets)

# Alt/null arrays per dataset
alt_arrays = {}
null_arrays = {}
for ds in datasets:
    sub = df_perturbed[df_perturbed["dataset"] == ds]
    alt_arrays[ds] = sub.loc[sub["distribution"] == "alt", "response"].values.astype(float)
    null_arrays[ds] = sub.loc[sub["distribution"] == "null", "response"].values.astype(float)

# ---------------------------------------------------------------------------
# Compute statistics for ridgeplot + scatter
# ---------------------------------------------------------------------------
seed_seq = np.random.SeedSequence(42)

records = []
for ds in datasets:
    ovl = overlap_coefficient(alt_arrays[ds], null_arrays[ds])
    child_rng = np.random.default_rng(seed_seq.spawn(1)[0])
    obs_mean, p_value, _, _ = bootstrap_mean_test(alt_arrays[ds], rng=child_rng)
    mean_sig = p_value < MEAN_ALPHA
    low_ovl = ovl < OVL_THRESHOLD
    if mean_sig and low_ovl:
        cat = siglow
    elif mean_sig and not low_ovl:
        cat = sighigh
    elif not mean_sig and low_ovl:
        cat = nonsiglow
    else:
        cat = nonsighigh
    records.append(dict(dataset=ds, obs_mean=obs_mean, p_value=p_value,
                        ovl=ovl, category=cat))

results = pd.DataFrame(records)

# ---------------------------------------------------------------------------
# Load agreement data (alt-only, full null)
# ---------------------------------------------------------------------------

def classify(row):
    sig = row["p_value"] < MEAN_ALPHA
    low_ovl = row["ovl"] < OVL_THRESHOLD
    if sig and low_ovl:
        return "Confident Yes"
    elif sig and not low_ovl:
        return "Analysis Failure"
    elif not sig and low_ovl:
        return "Low Signal"
    else:
        return "No Signal"


# Full-sample classification from stratified results
strat_df = pd.read_csv(BASE_DIR / "aggregated_results" / "sample_size_sensitivity.csv")
strat_df["category"] = strat_df.apply(classify, axis=1)
full_sample = (
    strat_df[strat_df["k"] == strat_df["k"].max()]
    .groupby("dataset")["category"]
    .agg(lambda s: s.mode().iloc[0])
    .to_dict()
)

# Alt-only sensitivity results
alt_only_df = pd.read_csv(
    BASE_DIR / "aggregated_results" / "sample_size_sensitivity_random_alt_only.csv"
)
alt_only_df["category"] = alt_only_df.apply(classify, axis=1)

alt_only_agreement = (
    alt_only_df[alt_only_df["n"] > 1]
    .groupby(["dataset", "n"])
    .apply(lambda g: (g["category"] == full_sample[g.name[0]]).mean())
    .reset_index(name="agreement")
)
n_filtered = sorted(alt_only_df.loc[alt_only_df["n"] > 1, "n"].unique())

# ---------------------------------------------------------------------------
# Figure: three panels
# ---------------------------------------------------------------------------
fig = plt.figure(figsize=(13.5, 5))
gs = GridSpec(1, 3, width_ratios=[0.75, 1.0, 0.75], wspace=0.32)
ax_ridge = fig.add_subplot(gs[0, 0])
ax_scatter = fig.add_subplot(gs[0, 1])
ax_agree = fig.add_subplot(gs[0, 2])

# --- Panel A: Ridgeplot ----------------------------------------------------
# Sort by alt mean (lowest at bottom, highest at top)
ridge_order = sorted(datasets, key=lambda ds: np.mean(alt_arrays[ds]), reverse=True)
for i, ds in enumerate(ridge_order):
    baseline = i * ROW_HEIGHT
    ax_ridge.axhline(baseline, color="gray", linewidth=0.4, zorder=0)

    for scores, color in [(null_arrays[ds], NULL_COLOR), (alt_arrays[ds], ALT_COLOR)]:
        if len(scores) < 2:
            continue
        kde = gaussian_kde(scores)
        dens = kde(X_GRID)
        dens = dens / dens.max() * RIDGE_SCALE
        ax_ridge.fill_between(X_GRID, baseline, baseline + dens,
                              color=color, alpha=0.45, linewidth=0)
        ax_ridge.plot(X_GRID, baseline + dens, color=color, linewidth=1.2)

    ax_ridge.text(-1, baseline + RIDGE_SCALE * 0.1, ds,
                  ha="right", va="bottom", fontsize=TICK_FS,
                  fontfamily="DejaVu Sans Mono")

ax_ridge.set_xlim(0, 100)
ax_ridge.set_ylim(-ROW_HEIGHT * 0.3, n_ds * ROW_HEIGHT + 0.2)
ax_ridge.set_xlabel("Response Score", fontsize=LABEL_FS)
ax_ridge.set_yticks([])
ax_ridge.tick_params(axis="x", labelsize=TICK_FS)
ax_ridge.legend(
    handles=[Patch(color=NULL_COLOR, alpha=0.7, label="Null"),
             Patch(color=ALT_COLOR, alpha=0.7, label="Alt")],
    loc="upper right", fontsize=LEGEND_FS,
)

# --- Panel B: Scatterplot --------------------------------------------------
texts = []
for cat in CATEGORIES:
    sub = results[results["category"] == cat]
    ax_scatter.scatter(sub["ovl"], sub["obs_mean"],
                       color=CAT_COLORS[cat], s=130, label=cat,
                       edgecolor="white", linewidth=0.6, zorder=3)
    for _, row in sub.iterrows():
        texts.append(ax_scatter.text(
            row["ovl"], row["obs_mean"], row["dataset"],
            fontsize=ANNOT_FS, fontfamily="DejaVu Sans Mono",
        ))

adjust_text(
    texts, ax=ax_scatter,
    expand=(1.3, 1.5),
    arrowprops=dict(arrowstyle="-", color="gray", lw=0.6),
)

ax_scatter.axhline(50, color="gray", linestyle="--", linewidth=0.8, zorder=1)
ax_scatter.axvline(OVL_THRESHOLD, color="gray", linestyle="--", linewidth=0.8, zorder=1)
ax_scatter.set_xlabel("Overlap Coefficient", fontsize=LABEL_FS)
ax_scatter.set_ylabel("Alt. Response Mean", fontsize=LABEL_FS)
ax_scatter.tick_params(labelsize=TICK_FS)
ax_scatter.set_xlim(-0.05, 1.05)
ax_scatter.legend(loc="upper right", fontsize=LEGEND_FS)

# --- Panel C: Agreement line plot ------------------------------------------
HIGHLIGHT = {"caschools": "sienna", "mortgage": "dodgerblue", "soccer": "darkorange"}
OTHER_COLOR = "0.35"

for ds in datasets:
    sub = alt_only_agreement[alt_only_agreement["dataset"] == ds]
    if ds in HIGHLIGHT:
        ax_agree.plot(sub["n"], sub["agreement"], marker="o", markersize=3,
                      color=HIGHLIGHT[ds], zorder=3)
    else:
        ax_agree.plot(sub["n"], sub["agreement"], marker="o", markersize=2,
                      color=OTHER_COLOR, linewidth=0.8, zorder=2)

# Inline labels for highlighted datasets, placed at a small n value
LABEL_POSITIONS = {"caschools": 7, "mortgage": 3, "soccer": 3}
for ds, color in HIGHLIGHT.items():
    sub = alt_only_agreement[alt_only_agreement["dataset"] == ds]
    label_n = LABEL_POSITIONS[ds]
    row = sub[sub["n"] == label_n]
    if not row.empty:
        y_val = row["agreement"].iloc[0]
        ax_agree.annotate(ds, xy=(label_n, y_val), xytext=(5, 0),
                          textcoords="offset points", fontsize=ANNOT_FS,
                          color="black", va="center",
                          fontfamily="DejaVu Sans Mono")

ax_agree.set_xscale("log")
ax_agree.set_xticks([2, 5, 10, 25, 50, 100])
ax_agree.get_xaxis().set_major_formatter(plt.ScalarFormatter())
ax_agree.minorticks_off()
ax_agree.tick_params(axis="x", labelsize=TICK_FS)
ax_agree.tick_params(axis="y", labelsize=TICK_FS)
ax_agree.set_xlabel("Runs per Dataset (Log)", fontsize=LABEL_FS)
ax_agree.set_ylabel("Agreement w/ full-sample class.", fontsize=LABEL_FS)
ax_agree.set_ylim(-0.02, 1.05)

# --- Panel labels ----------------------------------------------------------
for ax, label in [(ax_ridge, "(a)"), (ax_scatter, "(b)"), (ax_agree, "(c)")]:
    ax.text(0.5, -0.135, label, transform=ax.transAxes,
            ha="center", va="top", fontsize=LABEL_FS, fontweight="bold")

# --- Finalize --------------------------------------------------------------
sns.despine(ax=ax_ridge, left=True)
sns.despine(ax=ax_scatter)
sns.despine(ax=ax_agree)

stem = FIG_DIR / "three_panel_main_results"
fig.savefig(f"{stem}.png", dpi=600, bbox_inches="tight")
fig.savefig(f"{stem}.pdf", dpi=600, bbox_inches="tight")
print(f"Saved {stem}.png and {stem}.pdf")
