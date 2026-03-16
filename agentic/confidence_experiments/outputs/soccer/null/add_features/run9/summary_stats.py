import pandas as pd
import numpy as np
import statsmodels.api as sm

# Load dataset
path = "soccer.csv"
df = pd.read_csv(path)

df["skin_tone"] = df[["rater1", "rater2"]].mean(axis=1)
sub = df.dropna(subset=["skin_tone", "games", "redCards", "playerShort"])
sub = sub[sub["games"] > 0].copy()

# Poisson regression at dyad level with offset and cluster by player
X = sm.add_constant(sub[["skin_tone"]])
model = sm.GLM(
    sub["redCards"],
    X,
    family=sm.families.Poisson(),
    offset=np.log(sub["games"]),
)
res = model.fit(cov_type="cluster", cov_kwds={"groups": sub["playerShort"]})

coef = res.params["skin_tone"]
se = res.bse["skin_tone"]
p = res.pvalues["skin_tone"]
rr = np.exp(coef)

# Quartile comparison
q1 = sub["skin_tone"].quantile(0.25)
q3 = sub["skin_tone"].quantile(0.75)
sub_q = sub[(sub["skin_tone"] <= q1) | (sub["skin_tone"] >= q3)].copy()
sub_q["dark"] = (sub_q["skin_tone"] >= q3).astype(int)

X2 = sm.add_constant(sub_q[["dark"]])
model2 = sm.GLM(
    sub_q["redCards"],
    X2,
    family=sm.families.Poisson(),
    offset=np.log(sub_q["games"]),
)
res2 = model2.fit(cov_type="cluster", cov_kwds={"groups": sub_q["playerShort"]})

coef_d = res2.params["dark"]
se_d = res2.bse["dark"]
p_d = res2.pvalues["dark"]
rr_d = np.exp(coef_d)

summary = sub_q.groupby("dark").agg(games=("games", "sum"), redCards=("redCards", "sum"))
summary["rate"] = summary["redCards"] / summary["games"]
rate_light = summary.loc[0, "rate"]
rate_dark = summary.loc[1, "rate"]

print({
    "n_dyads": len(sub),
    "coef": coef,
    "se": se,
    "p": p,
    "rate_ratio": rr,
    "q1": q1,
    "q3": q3,
    "coef_dark": coef_d,
    "se_dark": se_d,
    "p_dark": p_d,
    "rate_ratio_dark": rr_d,
    "rate_light": rate_light,
    "rate_dark": rate_dark,
})
