import json
import numpy as np
import pandas as pd
import statsmodels.api as sm

DATA_PATH = "soccer.csv"

# Load data
# Note: dataset is wide but manageable; use low_memory to avoid dtype warnings
_df = pd.read_csv(DATA_PATH, low_memory=False)

# Map columns
red_cards = _df["feature16"]
matches = _df["feature9"]
skin1 = _df["feature18"]
skin2 = _df["feature19"]

# Compute mean skin tone; keep rows with at least one rating
skin_mean = pd.concat([skin1, skin2], axis=1).mean(axis=1, skipna=True)

# Build analysis frame
analysis = pd.DataFrame({
    "red_cards": red_cards,
    "matches": matches,
    "skin_mean": skin_mean,
})

# Basic cleaning
analysis = analysis.dropna(subset=["red_cards", "matches", "skin_mean"])
analysis = analysis[(analysis["matches"] > 0) & (analysis["red_cards"] >= 0)]

# Create binary dark/light using midpoint 0.5 on normalized 0-1 scale
analysis["tone_group"] = np.where(analysis["skin_mean"] > 0.5, "dark",
                                  np.where(analysis["skin_mean"] < 0.5, "light", "mid"))

# Summary stats
summary = {}
summary["n_total"] = int(len(analysis))
summary["n_dark"] = int((analysis["tone_group"] == "dark").sum())
summary["n_light"] = int((analysis["tone_group"] == "light").sum())
summary["n_mid"] = int((analysis["tone_group"] == "mid").sum())

# Rate per match by group
analysis["rate"] = analysis["red_cards"] / analysis["matches"]
rate_by_group = analysis.groupby("tone_group")["rate"].mean().to_dict()
summary["rate_by_group"] = {k: float(v) for k, v in rate_by_group.items()}

# Poisson regression: binary dark vs light (exclude mid)
binary = analysis[analysis["tone_group"].isin(["dark", "light"])].copy()
binary["dark"] = (binary["tone_group"] == "dark").astype(int)

X_bin = sm.add_constant(binary["dark"])
offset = np.log(binary["matches"])

poisson_bin = sm.GLM(binary["red_cards"], X_bin, family=sm.families.Poisson(), offset=offset)
poisson_bin_res = poisson_bin.fit(cov_type="HC1")

# IRR and p-value for dark
irr_dark = float(np.exp(poisson_bin_res.params["dark"]))
p_dark = float(poisson_bin_res.pvalues["dark"])

summary["poisson_binary"] = {
    "irr_dark_vs_light": irr_dark,
    "p_value": p_dark,
    "coef": float(poisson_bin_res.params["dark"]),
    "se": float(poisson_bin_res.bse["dark"]),
}

# Poisson regression: continuous skin tone
X_cont = sm.add_constant(analysis["skin_mean"])
offset_all = np.log(analysis["matches"])
poisson_cont = sm.GLM(analysis["red_cards"], X_cont, family=sm.families.Poisson(), offset=offset_all)
poisson_cont_res = poisson_cont.fit(cov_type="HC1")

irr_cont = float(np.exp(poisson_cont_res.params["skin_mean"]))
p_cont = float(poisson_cont_res.pvalues["skin_mean"])
summary["poisson_continuous"] = {
    "irr_per_unit": irr_cont,
    "p_value": p_cont,
    "coef": float(poisson_cont_res.params["skin_mean"]),
    "se": float(poisson_cont_res.bse["skin_mean"]),
}

# Negative binomial (robustness)
try:
    nb = sm.GLM(analysis["red_cards"], X_cont, family=sm.families.NegativeBinomial(), offset=offset_all)
    nb_res = nb.fit(cov_type="HC1")
    summary["nb_continuous"] = {
        "irr_per_unit": float(np.exp(nb_res.params["skin_mean"])),
        "p_value": float(nb_res.pvalues["skin_mean"]),
        "coef": float(nb_res.params["skin_mean"]),
        "se": float(nb_res.bse["skin_mean"]),
    }
except Exception as e:
    summary["nb_continuous_error"] = str(e)

# Save summary for inspection
with open("analysis_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print(json.dumps(summary, indent=2))
