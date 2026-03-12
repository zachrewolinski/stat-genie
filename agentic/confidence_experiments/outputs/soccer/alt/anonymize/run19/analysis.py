import json
import numpy as np
import pandas as pd
import statsmodels.api as sm

DF_PATH = "soccer.csv"

df = pd.read_csv(DF_PATH)

# Average skin tone rating across two raters (0 to 1 scale)
df["skin_mean"] = df[["feature18", "feature19"]].mean(axis=1)

# Define light vs dark groups using the extremes of the 5-point scale
# light: <= 0.25 (very light/light), dark: >= 0.75 (dark/very dark)
mask_light = df["skin_mean"] <= 0.25
mask_dark = df["skin_mean"] >= 0.75
subset = df[mask_light | mask_dark].copy()

subset = subset[subset["feature9"] > 0]
subset["dark"] = (subset["skin_mean"] >= 0.75).astype(int)

# Descriptive rates per match
summary = (
    subset.groupby("dark")[["feature16", "feature9"]]
    .sum()
    .rename(columns={"feature16": "red_cards", "feature9": "matches"})
)
summary["rate_per_match"] = summary["red_cards"] / summary["matches"]

# Poisson regression with log(matches) as offset
X = sm.add_constant(subset["dark"])
model = sm.GLM(
    subset["feature16"],
    X,
    family=sm.families.Poisson(),
    offset=np.log(subset["feature9"]),
)
res = model.fit(cov_type="HC3")

coef = res.params["dark"]
se = res.bse["dark"]
pval = res.pvalues["dark"]

irr = float(np.exp(coef))
ci_low = float(np.exp(coef - 1.96 * se))
ci_high = float(np.exp(coef + 1.96 * se))

output = {
    "n_rows": int(len(df)),
    "n_light_dark": int(len(subset)),
    "summary": summary.reset_index().to_dict(orient="records"),
    "poisson": {
        "coef_dark": float(coef),
        "se_dark": float(se),
        "pvalue_dark": float(pval),
        "irr_dark": irr,
        "irr_ci_low": ci_low,
        "irr_ci_high": ci_high,
    },
}

print(json.dumps(output, indent=2))
