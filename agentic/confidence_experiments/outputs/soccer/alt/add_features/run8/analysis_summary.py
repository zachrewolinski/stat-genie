import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from math import exp

path = "soccer.csv"
df = pd.read_csv(path)

df["skin_mean"] = df[["rater1", "rater2"]].mean(axis=1)

# summary counts
n_total = len(df)
n_skin = df["skin_mean"].notna().sum()

# bins
light = df["skin_mean"] <= 0.25
dark = df["skin_mean"] >= 0.75
mid = (df["skin_mean"] > 0.25) & (df["skin_mean"] < 0.75)

# rates
def rates(sub):
    red_sum = sub["redCards"].sum()
    games_sum = sub["games"].sum()
    rate = red_sum / games_sum if games_sum > 0 else np.nan
    any_rate = (sub["redCards"] > 0).mean()
    return red_sum, games_sum, rate, any_rate

light_stats = rates(df[light])
dark_stats = rates(df[dark])
mid_stats = rates(df[mid])

# poisson with skin_mean
poisson_df = df.dropna(subset=["redCards", "games", "skin_mean", "player"]).copy()
poisson_df = poisson_df[poisson_df["games"] > 0]

poisson_model = smf.glm(
    formula="redCards ~ skin_mean",
    data=poisson_df,
    family=sm.families.Poisson(),
    offset=np.log(poisson_df["games"])
).fit(cov_type="cluster", cov_kwds={"groups": poisson_df["player"]})

coef = poisson_model.params["skin_mean"]
se = poisson_model.bse["skin_mean"]
z = coef / se
p = poisson_model.pvalues["skin_mean"]
ci_low, ci_high = poisson_model.conf_int().loc["skin_mean"]
rr_full = exp(coef)
rr_low, rr_high = exp(ci_low), exp(ci_high)

# rate ratio for dark vs light via group model
poisson_df["skin_group"] = np.where(poisson_df["skin_mean"] <= 0.25, "light",
                           np.where(poisson_df["skin_mean"] >= 0.75, "dark", "mid"))
ld = poisson_df[poisson_df["skin_group"].isin(["light","dark"])]
poisson_ld = smf.glm(
    formula="redCards ~ C(skin_group)",
    data=ld,
    family=sm.families.Poisson(),
    offset=np.log(ld["games"])
).fit(cov_type="cluster", cov_kwds={"groups": ld["player"]})

coef_light = poisson_ld.params["C(skin_group)[T.light]"]
se_light = poisson_ld.bse["C(skin_group)[T.light]"]
p_light = poisson_ld.pvalues["C(skin_group)[T.light]"]
ci_light = poisson_ld.conf_int().loc["C(skin_group)[T.light]"]
rr_light_vs_dark = exp(coef_light)
rr_light_low, rr_light_high = exp(ci_light[0]), exp(ci_light[1])
rr_dark_vs_light = 1.0 / rr_light_vs_dark
rr_dark_low = 1.0 / rr_light_high
rr_dark_high = 1.0 / rr_light_low

# logistic any red
poisson_df["any_red"] = (poisson_df["redCards"] > 0).astype(int)
logit_model = smf.logit(
    formula="any_red ~ skin_mean + np.log(games)",
    data=poisson_df
).fit(disp=False)
logit_coef = logit_model.params["skin_mean"]
logit_se = logit_model.bse["skin_mean"]
logit_p = logit_model.pvalues["skin_mean"]
logit_ci = logit_model.conf_int().loc["skin_mean"]
logit_or = exp(logit_coef)
logit_or_low, logit_or_high = exp(logit_ci[0]), exp(logit_ci[1])

# effect for 0.5 increase (light to dark approx)
rr_half = exp(coef * 0.5)

summary = {
    "n_total": n_total,
    "n_with_skin": int(n_skin),
    "light_n": int(light.sum()),
    "dark_n": int(dark.sum()),
    "mid_n": int(mid.sum()),
    "light_redcards": float(light_stats[0]),
    "light_games": float(light_stats[1]),
    "light_rate": float(light_stats[2]),
    "light_any_rate": float(light_stats[3]),
    "dark_redcards": float(dark_stats[0]),
    "dark_games": float(dark_stats[1]),
    "dark_rate": float(dark_stats[2]),
    "dark_any_rate": float(dark_stats[3]),
    "poisson_coef": float(coef),
    "poisson_se": float(se),
    "poisson_p": float(p),
    "poisson_rr_full": float(rr_full),
    "poisson_rr_full_ci_low": float(rr_low),
    "poisson_rr_full_ci_high": float(rr_high),
    "poisson_rr_half": float(rr_half),
    "group_rr_light_vs_dark": float(rr_light_vs_dark),
    "group_rr_light_vs_dark_ci_low": float(rr_light_low),
    "group_rr_light_vs_dark_ci_high": float(rr_light_high),
    "group_rr_dark_vs_light": float(rr_dark_vs_light),
    "group_rr_dark_vs_light_ci_low": float(rr_dark_low),
    "group_rr_dark_vs_light_ci_high": float(rr_dark_high),
    "group_p_light_vs_dark": float(p_light),
    "logit_or": float(logit_or),
    "logit_or_ci_low": float(logit_or_low),
    "logit_or_ci_high": float(logit_or_high),
    "logit_p": float(logit_p),
}

for k, v in summary.items():
    print(f"{k}: {v}")
