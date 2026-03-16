import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.proportion import proportions_ztest
from scipy import stats

DATA_PATH = "mortgage.csv"

df = pd.read_csv(DATA_PATH)
df = df.replace([np.inf, -np.inf], np.nan)

# Define variables
female = df["feature2"]  # 1 if female
approved = df["feature14"]  # 1 if accepted

# Basic counts and rates
base = df.assign(female=female, approved=approved)
base = base.dropna(subset=["female", "approved"])

summary = (
    base.groupby("female")["approved"]
    .agg(["count", "mean", "sum"])
    .rename(columns={"count": "n", "mean": "approval_rate", "sum": "approved_count"})
)

# Two-proportion z-test for difference in approval rates
counts = summary["approved_count"].values
nobs = summary["n"].values
stat, pval = proportions_ztest(counts, nobs)

# Chi-square test of independence
contingency = pd.crosstab(base["female"], base["approved"])
chi2, chi2_p, _, _ = stats.chi2_contingency(contingency)

# Logistic regression with controls (exclude feature1 id-like and feature11 denied which is inverse of approval)
X_cols = [
    "feature2",
    "feature3",
    "feature4",
    "feature5",
    "feature6",
    "feature7",
    "feature8",
    "feature9",
    "feature10",
    "feature12",
    "feature13",
]
model_df = df[X_cols + ["feature14", "feature2"]].copy()
model_df = model_df.dropna()
X = model_df[X_cols]
X = sm.add_constant(X, has_constant="add")

glm = sm.GLM(model_df["feature14"], X, family=sm.families.Binomial()).fit()
coef = float(np.asarray(glm.params.loc["feature2"]).reshape(-1)[0])
pval_logit = float(np.asarray(glm.pvalues.loc["feature2"]).reshape(-1)[0])
conf_int = glm.conf_int().loc["feature2"].to_numpy().reshape(-1).tolist()

odds_ratio = float(np.exp(coef))
ci_or = [float(np.exp(conf_int[0])), float(np.exp(conf_int[1]))]

result = {
    "n_total": int(len(base)),
    "group_summary": {
        "male": {
            "n": int(summary.loc[0, "n"]),
            "approval_rate": float(summary.loc[0, "approval_rate"]),
        },
        "female": {
            "n": int(summary.loc[1, "n"]),
            "approval_rate": float(summary.loc[1, "approval_rate"]),
        },
    },
    "two_prop_z": {"z": float(stat), "p_value": float(pval)},
    "chi2": {"chi2": float(chi2), "p_value": float(chi2_p)},
    "logit": {
        "n_used": int(len(model_df)),
        "coef_female": float(coef),
        "odds_ratio": odds_ratio,
        "odds_ratio_ci": ci_or,
        "p_value": float(pval_logit),
    },
}

print(json.dumps(result, indent=2))
