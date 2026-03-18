import json
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm

# Load data
DATA_PATH = "affairs.csv"
INFO_PATH = "info.json"


df = pd.read_csv(DATA_PATH)

# Map columns using info.json
with open(INFO_PATH, "r") as f:
    info = json.load(f)

# Use metadata to understand columns
# feature2: affair frequency; feature6: children yes/no

# Clean/prepare
# Ensure feature6 is binary indicator (1 if children yes)
children = df["feature6"].astype(str).str.strip().str.lower()
child_yes = children.map({"yes": 1, "no": 0})

# Exclude rows with missing/unknown child indicator
valid = child_yes.notna()

# Affair frequency
affairs = df.loc[valid, "feature2"].astype(float)
child_yes = child_yes.loc[valid]

# Basic group stats
stats_by_group = (
    pd.DataFrame({"affairs": affairs, "child_yes": child_yes})
    .groupby("child_yes")["affairs"]
    .agg(["count", "mean", "median", "std"])
)

# Proportion with any affair
any_affair = (affairs > 0).astype(int)
prop_any = (
    pd.DataFrame({"any_affair": any_affair, "child_yes": child_yes})
    .groupby("child_yes")["any_affair"]
    .mean()
)

# Two-sample t-test (Welch) on affair frequency
child_affairs = affairs[child_yes == 1]
nochild_affairs = affairs[child_yes == 0]

# If either group empty, skip
if len(child_affairs) > 1 and len(nochild_affairs) > 1:
    t_stat, t_p = stats.ttest_ind(child_affairs, nochild_affairs, equal_var=False, nan_policy="omit")
else:
    t_stat, t_p = np.nan, np.nan

# Mann-Whitney U test (nonparametric)
if len(child_affairs) > 0 and len(nochild_affairs) > 0:
    try:
        u_stat, u_p = stats.mannwhitneyu(child_affairs, nochild_affairs, alternative="two-sided")
    except ValueError:
        u_stat, u_p = np.nan, np.nan
else:
    u_stat, u_p = np.nan, np.nan

# Logistic regression: any affair ~ children + controls
# Controls: gender (feature3), age (feature4), years married (feature5),
# religiousness (feature7), education (feature8), occupation (feature9),
# marriage rating (feature10)
model_df = df.loc[valid].copy()
model_df["child_yes"] = child_yes.values
model_df["any_affair"] = (model_df["feature2"].astype(float) > 0).astype(int)

# Encode gender as binary indicator (male=1, female=0)
model_df["male"] = model_df["feature3"].astype(str).str.strip().str.lower().map({"male": 1, "female": 0})

# Build design matrix
predictors = [
    "child_yes",
    "male",
    "feature4",
    "feature5",
    "feature7",
    "feature8",
    "feature9",
    "feature10",
]

X = model_df[predictors]
X = sm.add_constant(X)
y = model_df["any_affair"]

logit_result = None
logit_summary = {}
try:
    logit_model = sm.Logit(y, X, missing="drop")
    logit_result = logit_model.fit(disp=False)
    logit_summary = {
        "coef": float(logit_result.params["child_yes"]),
        "p_value": float(logit_result.pvalues["child_yes"]),
        "odds_ratio": float(np.exp(logit_result.params["child_yes"])),
    }
except Exception:
    logit_summary = {
        "coef": float("nan"),
        "p_value": float("nan"),
        "odds_ratio": float("nan"),
    }

# OLS regression: affair frequency ~ children + controls
ols_result = None
ols_summary = {}
try:
    ols_model = sm.OLS(model_df["feature2"].astype(float), X, missing="drop")
    ols_result = ols_model.fit()
    ols_summary = {
        "coef": float(ols_result.params["child_yes"]),
        "p_value": float(ols_result.pvalues["child_yes"]),
    }
except Exception:
    ols_summary = {
        "coef": float("nan"),
        "p_value": float("nan"),
    }

# Compute effect sizes
# Cohen's d
if len(child_affairs) > 1 and len(nochild_affairs) > 1:
    pooled_std = np.sqrt(
        ((child_affairs.var(ddof=1) + nochild_affairs.var(ddof=1)) / 2.0)
    )
    if pooled_std > 0:
        cohens_d = (child_affairs.mean() - nochild_affairs.mean()) / pooled_std
    else:
        cohens_d = np.nan
else:
    cohens_d = np.nan

results = {
    "stats_by_group": stats_by_group.to_dict(),
    "prop_any_affair": prop_any.to_dict(),
    "t_test": {"t_stat": float(t_stat), "p_value": float(t_p)},
    "mannwhitney": {"u_stat": float(u_stat), "p_value": float(u_p)},
    "ols": ols_summary,
    "logit": logit_summary,
    "cohens_d": float(cohens_d),
}

print(json.dumps(results, indent=2))
