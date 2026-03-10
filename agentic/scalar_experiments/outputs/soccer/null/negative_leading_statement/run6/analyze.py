import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
path = "soccer.csv"
df = pd.read_csv(path)

# Skin tone average
skin = df[["rater1", "rater2"]].mean(axis=1)
df = df.assign(skin=skin)

# Keep rows with skin and games
analysis = df.dropna(subset=["skin", "games", "redCards"]).copy()
analysis = analysis[analysis["games"] > 0]

# Descriptives: rate per game
analysis["red_rate"] = analysis["redCards"] / analysis["games"]
analysis["any_red"] = (analysis["redCards"] > 0).astype(int)
analysis["log_games"] = np.log(analysis["games"])

# Group by skin categories
# 5-point scale normalized to [0,1] in increments of 0.25
bins = [-0.01, 0.25, 0.5, 0.75, 1.01]
labels = ["very_light_light", "mid", "dark", "very_dark"]
analysis["skin_group"] = pd.cut(analysis["skin"], bins=bins, labels=labels)

desc = (
    analysis.groupby("skin_group")
    .agg(
        n=("redCards", "size"),
        total_red=("redCards", "sum"),
        total_games=("games", "sum"),
        mean_red_rate=("red_rate", "mean"),
        share_any_red=("any_red", "mean"),
    )
    .reset_index()
)

# Poisson GLM with exposure (games)
# Use robust SE (HC1)
poisson_model = smf.glm(
    formula="redCards ~ skin",
    data=analysis,
    family=sm.families.Poisson(),
    offset=analysis["log_games"],
).fit(cov_type="HC1")

# Negative binomial GLM for overdispersion check
nb_model = smf.glm(
    formula="redCards ~ skin",
    data=analysis,
    family=sm.families.NegativeBinomial(alpha=1.0),
    offset=analysis["log_games"],
).fit(cov_type="HC1")

# Logistic for any red with log games covariate
logit_model = smf.logit(
    formula="any_red ~ skin + log_games",
    data=analysis,
).fit(disp=False, cov_type="HC1")

# Overdispersion statistic for Poisson
mu = poisson_model.fittedvalues
var = analysis["redCards"].var()
mean = analysis["redCards"].mean()
overdisp_ratio = var / mean if mean > 0 else np.nan

# Correlation between skin and red rate
corr = analysis[["skin", "red_rate"]].corr().iloc[0,1]

# Save outputs for inspection
print("rows_used", len(analysis))
print("skin_unique", sorted(analysis["skin"].unique())[:10], "... total", analysis["skin"].nunique())
print("desc\n", desc)

print("poisson_coef", poisson_model.params["skin"], "p", poisson_model.pvalues["skin"])
print("poisson_ir", np.exp(poisson_model.params["skin"]))
print("nb_coef", nb_model.params["skin"], "p", nb_model.pvalues["skin"])
print("nb_ir", np.exp(nb_model.params["skin"]))
print("logit_coef", logit_model.params["skin"], "p", logit_model.pvalues["skin"])
print("logit_or", np.exp(logit_model.params["skin"]))
print("overdisp_ratio", overdisp_ratio)
print("corr_skin_red_rate", corr)
