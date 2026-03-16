import json
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

DATA_PATH = "soccer.csv"

# Load data
df = pd.read_csv(DATA_PATH)

# Compute skin tone mean when both raters available
# rater columns may have NaN
skin = df[["rater1", "rater2"]].mean(axis=1, skipna=False)
df = df.assign(skin_tone=skin)

# Define light/dark groups (exclude middle 0.5)
light_mask = df["skin_tone"].isin([0.0, 0.25])
dark_mask = df["skin_tone"].isin([0.75, 1.0])

# Subset to light/dark with non-missing skin and positive games
sub = df[(light_mask | dark_mask) & df["games"].notna() & (df["games"] > 0)].copy()
sub["dark"] = dark_mask.loc[sub.index].astype(int)

# Poisson regression with offset log(games)
sub["log_games"] = np.log(sub["games"])

# Use GLM Poisson
model = smf.glm("redCards ~ dark", data=sub, family=sm.families.Poisson(), offset=sub["log_games"]).fit()

coef = model.params["dark"]
se = model.bse["dark"]
pvalue = model.pvalues["dark"]
rate_ratio = float(np.exp(coef))

# Compute group rates per game
rate_light = (sub.loc[sub["dark"] == 0, "redCards"].sum() / sub.loc[sub["dark"] == 0, "games"].sum())
rate_dark = (sub.loc[sub["dark"] == 1, "redCards"].sum() / sub.loc[sub["dark"] == 1, "games"].sum())

# Also compute proportion of dyads with any red card
sub["any_red"] = (sub["redCards"] > 0).astype(int)
prop_light = sub.loc[sub["dark"] == 0, "any_red"].mean()
prop_dark = sub.loc[sub["dark"] == 1, "any_red"].mean()

results = {
    "n_rows": int(len(df)),
    "n_sub": int(len(sub)),
    "n_light": int((sub["dark"] == 0).sum()),
    "n_dark": int((sub["dark"] == 1).sum()),
    "rate_light": float(rate_light),
    "rate_dark": float(rate_dark),
    "prop_light": float(prop_light),
    "prop_dark": float(prop_dark),
    "poisson_coef": float(coef),
    "poisson_se": float(se),
    "poisson_pvalue": float(pvalue),
    "rate_ratio": float(rate_ratio),
}

print(json.dumps(results, indent=2))

# Create conclusion based on significance and effect size
# Likert scale: 0 strong no, 100 strong yes
# We'll use p-value and rate ratio: if p<0.05 and rate_ratio>1 -> yes with strength based on ratio
# if p>=0.05 -> no with mid-low value.

p = pvalue
rr = rate_ratio

if p < 0.05 and rr > 1:
    # Map rr to strength: rr 1.05 -> 60, rr 1.2 -> 70, rr 1.5 -> 80, rr 2 -> 90
    if rr < 1.1:
        score = 60
    elif rr < 1.25:
        score = 70
    elif rr < 1.6:
        score = 80
    else:
        score = 90
    verdict = "Yes"
elif p < 0.05 and rr < 1:
    # significant opposite direction
    score = 30
    verdict = "No (opposite direction)"
else:
    # not significant
    if rr > 1:
        score = 45
        verdict = "No (insufficient evidence)"
    else:
        score = 40
        verdict = "No (insufficient evidence)"

explanation = (
    f"Using dyads with clearly light (0 or 0.25) or dark (0.75 or 1.0) skin-tone ratings, "
    f"a Poisson regression of red-card counts with a log(games) offset estimates a rate ratio of {rr:.3f} "
    f"for dark vs light players (coef={coef:.3f}, SE={se:.3f}, p={p:.3g}). "
    f"The observed red-card rates per game were {rate_light:.4f} (light) vs {rate_dark:.4f} (dark), "
    f"and the share of dyads with any red card was {prop_light:.4f} (light) vs {prop_dark:.4f} (dark). "
    f"Based on statistical significance and effect size, the answer is {verdict}."
)

with open("conclusion.txt", "w") as f:
    json.dump({"response": int(score), "explanation": explanation}, f)
