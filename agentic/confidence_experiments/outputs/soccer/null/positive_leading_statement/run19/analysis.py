import json
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.rates import test_poisson_2indep, confint_poisson_2indep

# Load data
path = "soccer.csv"
df = pd.read_csv(path)

# Skin tone mean
skin = df["rater1"].astype(float).where(df["rater1"].notna())
skin2 = df["rater2"].astype(float).where(df["rater2"].notna())
df["skin_tone"] = pd.concat([skin, skin2], axis=1).mean(axis=1)

# Basic filters
analysis_df = df.copy()
analysis_df = analysis_df[analysis_df["games"].notna()]
analysis_df = analysis_df[analysis_df["games"] > 0]
analysis_df = analysis_df[analysis_df["skin_tone"].notna()].copy()

# Define skin tone groups
analysis_df["skin_group"] = pd.cut(
    analysis_df["skin_tone"],
    bins=[-0.01, 0.25, 0.75, 1.01],
    labels=["light", "mid", "dark"],
)

# Aggregate rates per group
agg = (
    analysis_df.groupby("skin_group")
    .agg(redCards_sum=("redCards", "sum"), games_sum=("games", "sum"), n=("redCards", "size"))
    .reset_index()
)
agg["rate_per_100_games"] = agg["redCards_sum"] / agg["games_sum"] * 100

# Light vs dark rate ratio test
light = agg[agg["skin_group"] == "light"].iloc[0]
dark = agg[agg["skin_group"] == "dark"].iloc[0]

test = test_poisson_2indep(
    count1=dark["redCards_sum"], exposure1=dark["games_sum"],
    count2=light["redCards_sum"], exposure2=light["games_sum"],
    method="score"
)

ci_low, ci_high = confint_poisson_2indep(
    count1=dark["redCards_sum"], exposure1=dark["games_sum"],
    count2=light["redCards_sum"], exposure2=light["games_sum"],
    method="score"
)

# Prepare regression data
reg_df = analysis_df.copy()

# Parse age from birthday (assume season 2012-2013, use 2013 as reference year)
# birthday format dd.mm.yyyy
birth_year = reg_df["birthday"].astype(str).str[-4:]
reg_df["birth_year"] = pd.to_numeric(birth_year, errors="coerce")
reg_df["age"] = 2013 - reg_df["birth_year"]

# Select columns and drop missing
cols = [
    "redCards", "games", "skin_tone", "height", "weight", "age",
    "position", "leagueCountry"
]
reg_df = reg_df[cols].dropna().copy()

# Ensure categorical types
reg_df["position"] = reg_df["position"].astype("category")
reg_df["leagueCountry"] = reg_df["leagueCountry"].astype("category")

# Poisson regression with offset for games
reg_df["log_games"] = np.log(reg_df["games"])

formula = "redCards ~ skin_tone + height + weight + age + C(position) + C(leagueCountry)"
model = smf.glm(
    formula=formula,
    data=reg_df,
    family=sm.families.Poisson(),
    offset=reg_df["log_games"],
)
# Cluster-robust SE by position to account for heterogeneity
res = model.fit(cov_type="cluster", cov_kwds={"groups": reg_df["position"]})

# Overdispersion check
mu = res.mu
overdispersion = ((reg_df["redCards"] - mu) ** 2).sum() / res.df_resid

# Negative binomial regression (sensitivity)
nb_model = smf.glm(
    formula=formula,
    data=reg_df,
    family=sm.families.NegativeBinomial(alpha=1.0),
    offset=reg_df["log_games"],
)
nb_res = nb_model.fit(cov_type="cluster", cov_kwds={"groups": reg_df["position"]})

# Extract key results
poisson_coef = res.params["skin_tone"]
poisson_se = res.bse["skin_tone"]
poisson_p = res.pvalues["skin_tone"]
poisson_irr = float(np.exp(poisson_coef))

nb_coef = nb_res.params["skin_tone"]
nb_se = nb_res.bse["skin_tone"]
nb_p = nb_res.pvalues["skin_tone"]
nb_irr = float(np.exp(nb_coef))

# Build output summary
summary = {
    "n_rows": int(len(analysis_df)),
    "rate_table": agg.to_dict(orient="records"),
    "light_vs_dark_rate_ratio": {
        "rate_ratio": float(test.ratio),
        "p_value": float(test.pvalue),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
    },
    "poisson": {
        "coef": float(poisson_coef),
        "se": float(poisson_se),
        "p_value": float(poisson_p),
        "irr": poisson_irr,
    },
    "neg_binom": {
        "coef": float(nb_coef),
        "se": float(nb_se),
        "p_value": float(nb_p),
        "irr": nb_irr,
    },
    "overdispersion": float(overdispersion),
}

with open("analysis_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print(json.dumps(summary, indent=2))
