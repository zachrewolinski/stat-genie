import json

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm


df = pd.read_csv("affairs.csv")

cols = {
    "feature2": "affairs",
    "feature6": "children",
    "feature3": "gender",
    "feature4": "age",
    "feature5": "years_married",
    "feature7": "religious",
    "feature8": "education",
    "feature9": "occupation",
    "feature10": "marriage_rating",
}

df = df.rename(columns=cols)

if df["children"].dtype != object:
    df["children"] = df["children"].astype(str)

df["children"] = df["children"].str.lower().str.strip()

df["any_affair"] = (df["affairs"] > 0).astype(int)

df["children_yes"] = (df["children"] == "yes").astype(int)

yes_aff = df.loc[df["children_yes"] == 1, "affairs"]
no_aff = df.loc[df["children_yes"] == 0, "affairs"]

group_stats = df.groupby("children")["affairs"].agg(["count", "mean", "median", "std"])
prop_any = df.groupby("children")["any_affair"].mean()

ttest = stats.ttest_ind(yes_aff, no_aff, equal_var=False, nan_policy="omit")
mw = stats.mannwhitneyu(yes_aff, no_aff, alternative="two-sided")


def cohens_d(x, y):
    x = np.asarray(x)
    y = np.asarray(y)
    nx = len(x)
    ny = len(y)
    vx = np.nanvar(x, ddof=1)
    vy = np.nanvar(y, ddof=1)
    s = np.sqrt(((nx - 1) * vx + (ny - 1) * vy) / (nx + ny - 2))
    if s == 0:
        return np.nan
    return (np.nanmean(x) - np.nanmean(y)) / s


cd = cohens_d(yes_aff, no_aff)

df["gender"] = df["gender"].astype(str).str.lower().str.strip()

df["male"] = (df["gender"] == "male").astype(int)

logit = sm.Logit(df["any_affair"], sm.add_constant(df["children_yes"])).fit(disp=False)

adj_X = sm.add_constant(
    df[
        [
            "children_yes",
            "male",
            "age",
            "years_married",
            "religious",
            "education",
            "occupation",
            "marriage_rating",
        ]
    ]
)

adj_logit = sm.Logit(df["any_affair"], adj_X).fit(disp=False)

ols = sm.OLS(df["affairs"], sm.add_constant(df[["children_yes"]])).fit(cov_type="HC1")

adj_ols_X = sm.add_constant(
    df[
        [
            "children_yes",
            "male",
            "age",
            "years_married",
            "religious",
            "education",
            "occupation",
            "marriage_rating",
        ]
    ]
)

adj_ols = sm.OLS(df["affairs"], adj_ols_X).fit(cov_type="HC1")

mean_no = group_stats.loc["no", "mean"]
mean_yes = group_stats.loc["yes", "mean"]
median_no = group_stats.loc["no", "median"]
median_yes = group_stats.loc["yes", "median"]
prop_no = prop_any.loc["no"]
prop_yes = prop_any.loc["yes"]

explanation = (
    "Comparing couples with and without children shows virtually no difference in extramarital "
    f"affairs. Mean affairs are {mean_yes:.3f} for those with children vs {mean_no:.3f} without, "
    f"and medians are {median_yes:.3f} vs {median_no:.3f}. The Welch t-test is not significant "
    f"(p={ttest.pvalue:.3f}) and the Mann-Whitney test also shows no difference (p={mw.pvalue:.3f}); "
    f"the effect size is essentially zero (Cohen's d={cd:.3f}). The share reporting any affair is "
    f"{prop_yes*100:.1f}% with children vs {prop_no*100:.1f}% without, a {((prop_yes-prop_no)*100):.1f} "
    "percentage-point difference. Logistic regression on any affair finds no association with "
    f"children (p={logit.pvalues['children_yes']:.3f}), and the result remains non-significant with "
    f"controls (p={adj_logit.pvalues['children_yes']:.3f}). OLS on affair frequency similarly shows no "
    f"significant effect of children unadjusted (p={ols.pvalues['children_yes']:.3f}) or adjusted "
    f"(p={adj_ols.pvalues['children_yes']:.3f}). Overall, the data do not support that having children "
    "decreases engagement in extramarital affairs."
)

result = {"response": 15, "explanation": explanation}

with open("conclusion.txt", "w", encoding="utf-8") as f:
    json.dump(result, f)
