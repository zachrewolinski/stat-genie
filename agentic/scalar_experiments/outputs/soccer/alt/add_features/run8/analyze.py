import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
path = "soccer.csv"
df = pd.read_csv(path)

print("rows", len(df), "cols", len(df.columns))
print("columns", df.columns.tolist())

# Skin tone mean
if "rater1" in df.columns and "rater2" in df.columns:
    df["skin_mean"] = df[["rater1", "rater2"]].mean(axis=1)
else:
    raise SystemExit("Missing rater columns")

print("skin_mean unique", df["skin_mean"].dropna().unique()[:10])
print("skin_mean value counts", df["skin_mean"].value_counts(dropna=False).head(10))

# Red cards stats
print("redCards stats", df["redCards"].describe())
print("redCards nonzero", (df["redCards"]>0).mean())

# Basic rate by skin bin (light vs dark) with threshold
# define light: <=0.25, dark: >=0.75, exclude middle (0.5)
light = df["skin_mean"] <= 0.25
dark = df["skin_mean"] >= 0.75
mid = (df["skin_mean"] > 0.25) & (df["skin_mean"] < 0.75)

print("counts light/dark/mid", light.sum(), dark.sum(), mid.sum())

# compute red card rate per game for light/dark
for label, mask in [("light", light), ("dark", dark), ("mid", mid)]:
    sub = df[mask]
    if len(sub) == 0:
        continue
    rate = sub["redCards"].sum() / sub["games"].sum()
    any_rate = (sub["redCards"]>0).mean()
    print(label, "redcards per game", rate, "any redcard", any_rate)

# Poisson regression with offset log(games)
# Use skin_mean as continuous predictor
poisson_df = df.dropna(subset=["redCards", "games", "skin_mean", "player"])
# Avoid zero games
poisson_df = poisson_df[poisson_df["games"] > 0]

# Add small constant? redCards can be zero, OK.
# Fit GLM Poisson with log(games) offset
poisson_model = smf.glm(
    formula="redCards ~ skin_mean",
    data=poisson_df,
    family=sm.families.Poisson(),
    offset=np.log(poisson_df["games"])
).fit(cov_type="cluster", cov_kwds={"groups": poisson_df["player"]})

print(poisson_model.summary())

# Also logistic model for any red card with games control
poisson_df["any_red"] = (poisson_df["redCards"] > 0).astype(int)
logit_model = smf.logit(
    formula="any_red ~ skin_mean + np.log(games)",
    data=poisson_df
).fit(disp=False)

print(logit_model.summary())

# Exponentiated coefficients
print("Poisson exp(beta) per 1.0 skin_mean", np.exp(poisson_model.params["skin_mean"]))
print("Logit OR per 1.0 skin_mean", np.exp(logit_model.params["skin_mean"]))

# Also compare light vs dark using poisson with indicator
poisson_df = poisson_df.copy()
poisson_df["skin_group"] = np.where(poisson_df["skin_mean"] <= 0.25, "light",
                           np.where(poisson_df["skin_mean"] >= 0.75, "dark", "mid"))

# Use only light and dark
ld = poisson_df[poisson_df["skin_group"].isin(["light","dark"])]
poisson_ld = smf.glm(
    formula="redCards ~ C(skin_group)",
    data=ld,
    family=sm.families.Poisson(),
    offset=np.log(ld["games"])
).fit(cov_type="cluster", cov_kwds={"groups": ld["player"]})
print(poisson_ld.summary())

# print rate ratio dark vs light
print("Rate ratio dark vs light", np.exp(poisson_ld.params.get("C(skin_group)[T.dark]", np.nan)))
