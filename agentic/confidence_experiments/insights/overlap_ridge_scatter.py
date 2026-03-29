"""Side-by-side ridgeplot + scatterplot for KDE overlap analysis."""

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
# Style
# ---------------------------------------------------------------------------
sns.set_style("whitegrid")

NULL_COLOR = "#008080"
ALT_COLOR = "#FF7F50"

CAT_COLORS = {
    "Confident Yes": "seagreen",
    "Low Signal": "goldenrod",
    "Analysis Failure": "mediumpurple",
    "No Signal": "lightcoral",
}
CATEGORIES = list(CAT_COLORS)

OVL_THRESHOLD = 0.2
MEAN_ALPHA = 0.05

# Ridge params
X_GRID = np.linspace(0, 100, 500)
ROW_HEIGHT = 1.0
RIDGE_SCALE = 0.75

# Font sizes — ridge
RIDGE_LABEL_FS = 24
RIDGE_TICK_FS = 14
RIDGE_LEGEND_FS = 14

# Font sizes — scatter
SCATTER_LABEL_FS = 20
SCATTER_TICK_FS = 16
SCATTER_ANNOT_FS = 16
SCATTER_LEGEND_FS = 16

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

# Pre-extract alt/null arrays per dataset
alt_arrays = {}
null_arrays = {}
for ds in datasets:
    sub = df_perturbed[df_perturbed["dataset"] == ds]
    alt_arrays[ds] = sub.loc[sub["distribution"] == "alt", "response"].values.astype(float)
    null_arrays[ds] = sub.loc[sub["distribution"] == "null", "response"].values.astype(float)

# ---------------------------------------------------------------------------
# Compute statistics
# ---------------------------------------------------------------------------
seed_seq = np.random.SeedSequence(42)

records = []
for i, ds in enumerate(datasets):
    ovl = overlap_coefficient(alt_arrays[ds], null_arrays[ds])
    child_rng = np.random.default_rng(seed_seq.spawn(1)[0])
    obs_mean, p_value, _, _ = bootstrap_mean_test(alt_arrays[ds], rng=child_rng)
    mean_sig = p_value < MEAN_ALPHA
    low_ovl = ovl < OVL_THRESHOLD
    if mean_sig and low_ovl:
        cat = "Confident Yes"
    elif mean_sig and not low_ovl:
        cat = "Analysis Failure"
    elif not mean_sig and low_ovl:
        cat = "Low Signal"
    else:
        cat = "No Signal"
    records.append(dict(dataset=ds, obs_mean=obs_mean, p_value=p_value,
                        ovl=ovl, category=cat))

results = pd.DataFrame(records)

# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
fig = plt.figure(figsize=(20, 6.9))
gs = GridSpec(1, 2, width_ratios=[0.85, 1.15], wspace=0.3)
ax_ridge = fig.add_subplot(gs[0, 0])
ax_scatter = fig.add_subplot(gs[0, 1])

# --- Ridgeplot (left) -----------------------------------------------------
for i, ds in enumerate(reversed(datasets)):
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
                  ha="right", va="bottom", fontsize=RIDGE_TICK_FS,
                  fontfamily="DejaVu Sans Mono")

ax_ridge.set_xlim(0, 100)
ax_ridge.set_ylim(-ROW_HEIGHT * 0.3, n_ds * ROW_HEIGHT + 0.2)
ax_ridge.set_xlabel("Response Score", fontsize=RIDGE_LABEL_FS)
ax_ridge.set_yticks([])
ax_ridge.tick_params(axis="x", labelsize=RIDGE_TICK_FS)
ax_ridge.legend(
    handles=[Patch(color=NULL_COLOR, alpha=0.7, label="Null"),
             Patch(color=ALT_COLOR, alpha=0.7, label="Alt")],
    loc="upper right", fontsize=RIDGE_LEGEND_FS,
)

# --- Scatterplot (right) --------------------------------------------------
texts = []
for cat in CATEGORIES:
    sub = results[results["category"] == cat]
    ax_scatter.scatter(sub["ovl"], sub["obs_mean"],
                       color=CAT_COLORS[cat], s=130, label=cat,
                       edgecolor="white", linewidth=0.6, zorder=3)
    for _, row in sub.iterrows():
        texts.append(ax_scatter.text(
            row["ovl"], row["obs_mean"], row["dataset"],
            fontsize=SCATTER_ANNOT_FS, fontfamily="DejaVu Sans Mono",
        ))

adjust_text(
    texts, ax=ax_scatter,
    expand=(1.3, 1.5),
    arrowprops=dict(arrowstyle="-", color="gray", lw=0.6),
)

ax_scatter.axhline(50, color="gray", linestyle="--", linewidth=0.8, zorder=1)
ax_scatter.axvline(OVL_THRESHOLD, color="gray", linestyle="--", linewidth=0.8, zorder=1)
ax_scatter.set_xlabel("Overlap Coefficient", fontsize=SCATTER_LABEL_FS)
ax_scatter.set_ylabel("Alt. Response Mean", fontsize=SCATTER_LABEL_FS)
ax_scatter.tick_params(labelsize=SCATTER_TICK_FS)
ax_scatter.set_xlim(-0.05, 1.05)
ax_scatter.legend(loc="upper right", fontsize=SCATTER_LEGEND_FS)

# --- Finalize --------------------------------------------------------------
sns.despine(ax=ax_ridge, left=True)
sns.despine(ax=ax_scatter)

stem = FIG_DIR / "overlap_ridge_scatter"
fig.savefig(f"{stem}.png", dpi=600, bbox_inches="tight")
fig.savefig(f"{stem}.pdf", dpi=600, bbox_inches="tight")
print(f"Saved {stem}.png and {stem}.pdf")
