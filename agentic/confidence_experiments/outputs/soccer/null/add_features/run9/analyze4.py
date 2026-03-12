import pandas as pd
import numpy as np
import statsmodels.api as sm

# Load dataset
path = "soccer.csv"
df = pd.read_csv(path)

# Skin tone
skin = df[["rater1", "rater2"]].mean(axis=1)
df = df.assign(skin_tone=skin)

sub = df.dropna(subset=["skin_tone", "games", "redCards", "playerShort"])
sub = sub[sub["games"] > 0].copy()

# Poisson regression at dyad level with offset
X = sm.add_constant(sub[["skin_tone"]])
model = sm.GLM(
    sub["redCards"],
    X,
    family=sm.families.Poisson(),
    offset=np.log(sub["games"]),
)
res = model.fit(cov_type="cluster", cov_kwds={"groups": sub["playerShort"]})
print(res.summary())

# Also compare top vs bottom quartile of skin tone (player-level quartiles but applied to dyads)
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
print(res2.summary())

# Simple rate comparison for quartiles
summary = sub_q.groupby("dark").agg(games=("games", "sum"), redCards=("redCards", "sum"))
summary["rate"] = summary["redCards"] / summary["games"]
print(summary)
