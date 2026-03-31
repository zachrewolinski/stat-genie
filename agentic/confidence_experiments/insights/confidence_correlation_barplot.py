"""Barplot of Spearman rho between stated confidence and empirical exceedance."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

sns.set_style("white")

NULL_COLOR = "0.6"
ALT_COLOR = "#FF7F50"

BASE_DIR = Path("..").resolve()
CSV_PATH = BASE_DIR / "aggregated_results" / "aggregated_results.csv"
FIG_DIR = Path("figures")
FIG_DIR.mkdir(exist_ok=True)

df = pd.read_csv(CSV_PATH, keep_default_na=False)
df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce")
df = df[
    (df["distribution"].isin(["null", "alt"]))
    & (df["perturbation"] != "none")
].dropna(subset=["confidence"]).copy()

# empirical exceedance within (dataset, distribution)
def broad_exceedance(group):
    r = group["response"].values
    n = len(r)
    return pd.Series(
        [(r > v).sum() / (n - 1) for v in r],
        index=group.index,
    )

df["emp_exceedance"] = df.groupby(
    ["dataset", "distribution"], group_keys=False
).apply(broad_exceedance)

rhos = {}
for dist in ["alt", "null"]:
    sub = df[df["distribution"] == dist]
    rho, _ = stats.spearmanr(sub["confidence"], sub["emp_exceedance"])
    rhos[dist] = rho

# plot
fig, ax = plt.subplots(figsize=(2.4, 2.8))
bars = ax.bar(
    ["Alt", "Null"],
    [rhos["alt"], rhos["null"]],
    color=[ALT_COLOR, NULL_COLOR],
    width=0.7,
    edgecolor="white",
)

for bar, val in zip(bars, [rhos["alt"], rhos["null"]]):
    ax.text(bar.get_x() + bar.get_width() / 2, val + 0.01,
            f"{val:.3f}", ha="center", va="bottom", fontsize=10)

ax.set_ylabel("Correlation")
ax.set_ylim(0, 1)
ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax.yaxis.grid(True, linewidth=0.5, color="0.85")
ax.set_axisbelow(True)
sns.despine(ax=ax)
plt.tight_layout()

stem = FIG_DIR / "confidence_exceedance_correlation"
fig.savefig(f"{stem}.pdf", bbox_inches="tight")
fig.savefig(f"{stem}.png", dpi=300, bbox_inches="tight")
print(f"Saved {stem}.pdf and {stem}.png")
print(f"  Alt rho = {rhos['alt']:.3f}, Null rho = {rhos['null']:.3f}")
