import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

DATA_PATH = "soccer.csv"

df = pd.read_csv(DATA_PATH)

# Compute skin tone as mean of available rater scores
skin = df[["feature18", "feature19"]].mean(axis=1, skipna=True)

df = df.copy()
_df = df.copy()
_df["skin_tone"] = skin

# Keep rows with necessary info
_df = _df[_df["skin_tone"].notna()]
_df = _df[_df["feature16"].notna()]
_df = _df[_df["feature9"].notna()]
_df = _df[_df["feature9"] > 0]

# Basic counts
n_total = len(_df)

# Define light vs dark using extremes: light <=0.25, dark >=0.75
_df["tone_group"] = pd.cut(
    _df["skin_tone"],
    bins=[-np.inf, 0.25, 0.75, np.inf],
    labels=["light", "mid", "dark"],
)

# Compute red card rate per game by group
rate_by_group = (
    _df.groupby("tone_group")
    .apply(lambda g: g["feature16"].sum() / g["feature9"].sum())
    .rename("red_per_game")
)

# Poisson regression: red cards with offset log(games)
# predictor: continuous skin_tone
_poisson_df = _df.copy()
_poisson_df["log_games"] = np.log(_poisson_df["feature9"])

model = sm.GLM(
    _poisson_df["feature16"],
    sm.add_constant(_poisson_df["skin_tone"]),
    family=sm.families.Poisson(),
    offset=_poisson_df["log_games"],
)

poisson_res = model.fit(cov_type="HC0")

# Negative binomial as robustness (if it converges)
try:
    nb_model = sm.GLM(
        _poisson_df["feature16"],
        sm.add_constant(_poisson_df["skin_tone"]),
        family=sm.families.NegativeBinomial(alpha=1.0),
        offset=_poisson_df["log_games"],
    )
    nb_res = nb_model.fit(cov_type="HC0")
except Exception as e:
    nb_res = None

# Logistic regression: any red card in dyad
_poisson_df["any_red"] = (_poisson_df["feature16"] > 0).astype(int)
logit_model = sm.Logit(
    _poisson_df["any_red"],
    sm.add_constant(_poisson_df["skin_tone"]),
)
logit_res = logit_model.fit(disp=False)

# Prepare summary for printing
summary = {
    "n_rows_used": int(n_total),
    "tone_group_counts": _df["tone_group"].value_counts().to_dict(),
    "red_rate_per_game": rate_by_group.to_dict(),
    "poisson_coef": float(poisson_res.params["skin_tone"]),
    "poisson_p": float(poisson_res.pvalues["skin_tone"]),
    "poisson_ir" : float(np.exp(poisson_res.params["skin_tone"]))
}

if nb_res is not None:
    summary.update({
        "nb_coef": float(nb_res.params["skin_tone"]),
        "nb_p": float(nb_res.pvalues["skin_tone"]),
        "nb_ir": float(np.exp(nb_res.params["skin_tone"]))
    })
else:
    summary.update({"nb_coef": None, "nb_p": None, "nb_ir": None})

summary.update({
    "logit_coef": float(logit_res.params["skin_tone"]),
    "logit_p": float(logit_res.pvalues["skin_tone"]),
    "logit_or": float(np.exp(logit_res.params["skin_tone"]))
})

print(json.dumps(summary, indent=2))
