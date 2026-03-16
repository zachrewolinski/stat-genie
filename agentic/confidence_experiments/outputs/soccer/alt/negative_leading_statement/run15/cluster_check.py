import pandas as pd
import numpy as np
import statsmodels.api as sm


df = pd.read_csv("soccer.csv")
df["skin_tone"] = df[["rater1", "rater2"]].mean(axis=1)
df = df[(~df["skin_tone"].isna()) & (df["games"] > 0)].copy()
df["skin_group_extreme"] = pd.cut(
    df["skin_tone"], bins=[-np.inf, 0.25, 0.75, np.inf], labels=["light", "medium", "dark"]
)

extreme_df = df[df["skin_group_extreme"].isin(["light", "dark"])].copy()
extreme_df["is_dark"] = (extreme_df["skin_group_extreme"] == "dark").astype(int)
extreme_df["intercept"] = 1.0

poisson_extreme = sm.GLM(
    extreme_df["redCards"],
    extreme_df[["intercept", "is_dark"]],
    family=sm.families.Poisson(),
    offset=np.log(extreme_df["games"]),
).fit()

print("standard pvalues", poisson_extreme.pvalues)

try:
    robust = poisson_extreme.get_robustcov_results(cov_type="cluster", groups=extreme_df["playerShort"])
    print("cluster pvalues", robust.pvalues)
    print("cluster ci", robust.conf_int())
except Exception as e:
    print("cluster error", e)
