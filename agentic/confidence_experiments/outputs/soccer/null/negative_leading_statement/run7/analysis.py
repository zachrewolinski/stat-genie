import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf


df = pd.read_csv("soccer.csv")

# Skin tone scale is 0-1 in 5 steps; use average of two raters.
df["skin_tone"] = df[["rater1", "rater2"]].mean(axis=1)

# Keep dyads with games > 0 and known skin tone
base = df[(df["games"] > 0) & (df["skin_tone"].notna())].copy()

# Define light/dark groups (extremes of the 5-point scale)
base["dark"] = base["skin_tone"] >= 0.75
base["light"] = base["skin_tone"] <= 0.25
main = base[base["dark"] | base["light"]].copy()
main["dark_indicator"] = main["dark"].astype(int)

# Summary rates
summary = (
    main.groupby("dark_indicator")
    .agg(dyads=("redCards", "size"), games=("games", "sum"), redcards=("redCards", "sum"))
)
summary["rate_per_game"] = summary["redcards"] / summary["games"]

# Poisson regression with exposure offset
main["log_games"] = np.log(main["games"])
model = smf.glm(
    "redCards ~ dark_indicator",
    data=main,
    family=sm.families.Poisson(),
    offset=main["log_games"],
).fit(cov_type="HC1")

# Continuous skin tone model (all observations)
base["log_games"] = np.log(base["games"])
model_cont = smf.glm(
    "redCards ~ skin_tone",
    data=base,
    family=sm.families.Poisson(),
    offset=base["log_games"],
).fit(cov_type="HC1")

# Adjusted model with basic covariates (position, leagueCountry, height, weight)
# Use only rows with non-missing covariates
adj_cols = ["position", "leagueCountry", "height", "weight"]
adj = main.dropna(subset=adj_cols).copy()
if len(adj) > 0:
    adj["log_games"] = np.log(adj["games"])
    model_adj = smf.glm(
        "redCards ~ dark_indicator + C(position) + C(leagueCountry) + height + weight",
        data=adj,
        family=sm.families.Poisson(),
        offset=adj["log_games"],
    ).fit(cov_type="HC1")
else:
    model_adj = None
    robust_adj = None

# Print key results
print("SUMMARY (light=0, dark=1):")
print(summary)
print("\nPOISSON (dark_indicator):")
print(model.summary().tables[1])
print("\nPOISSON (skin_tone continuous):")
print(model_cont.summary().tables[1])
if model_adj is not None:
    print("\nPOISSON ADJUSTED (dark_indicator + covariates):")
    print(model_adj.summary().tables[1])

# Convenience numbers
coef = model.params.get("dark_indicator", np.nan)
se = model.bse.get("dark_indicator", np.nan)
pval = model.pvalues.get("dark_indicator", np.nan)
rr = np.exp(coef) if pd.notna(coef) else np.nan
ci_low = np.exp(coef - 1.96 * se) if pd.notna(coef) else np.nan
ci_high = np.exp(coef + 1.96 * se) if pd.notna(coef) else np.nan

print("\nDARK VS LIGHT RATE RATIO (unadjusted, Poisson):")
print({"rate_ratio": rr, "ci_low": ci_low, "ci_high": ci_high, "p_value": pval})

coef_c = model_cont.params.get("skin_tone", np.nan)
se_c = model_cont.bse.get("skin_tone", np.nan)
pval_c = model_cont.pvalues.get("skin_tone", np.nan)
rr_c = np.exp(coef_c) if pd.notna(coef_c) else np.nan
ci_low_c = np.exp(coef_c - 1.96 * se_c) if pd.notna(coef_c) else np.nan
ci_high_c = np.exp(coef_c + 1.96 * se_c) if pd.notna(coef_c) else np.nan

print("\nSKIN TONE CONTINUOUS RATE RATIO (per 1.0 increase):")
print({"rate_ratio": rr_c, "ci_low": ci_low_c, "ci_high": ci_high_c, "p_value": pval_c})

if model_adj is not None:
    coef_a = model_adj.params.get("dark_indicator", np.nan)
    se_a = model_adj.bse.get("dark_indicator", np.nan)
    pval_a = model_adj.pvalues.get("dark_indicator", np.nan)
    rr_a = np.exp(coef_a) if pd.notna(coef_a) else np.nan
    ci_low_a = np.exp(coef_a - 1.96 * se_a) if pd.notna(coef_a) else np.nan
    ci_high_a = np.exp(coef_a + 1.96 * se_a) if pd.notna(coef_a) else np.nan
    print("\nADJUSTED DARK VS LIGHT RATE RATIO:")
    print({"rate_ratio": rr_a, "ci_low": ci_low_a, "ci_high": ci_high_a, "p_value": pval_a})
