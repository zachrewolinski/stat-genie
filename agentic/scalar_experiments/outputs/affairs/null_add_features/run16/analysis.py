import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf


def cohen_d(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    na = a.size
    nb = b.size
    if na < 2 or nb < 2:
        return np.nan
    va = a.var(ddof=1)
    vb = b.var(ddof=1)
    s = np.sqrt(((na - 1) * va + (nb - 1) * vb) / (na + nb - 2))
    if s == 0:
        return 0.0
    return (a.mean() - b.mean()) / s


df = pd.read_csv("affairs.csv")

# Basic cleaning
# Ensure children is categorical yes/no
if df["children"].dtype != object:
    df["children"] = df["children"].astype(str)

df = df[df["children"].isin(["yes", "no"])].copy()

df["children_yes"] = (df["children"] == "yes").astype(int)

df["any_affair"] = (df["affairs"] > 0).astype(int)

# Group stats
grp = df.groupby("children")
mean_affairs = grp["affairs"].mean()
prop_any = grp["any_affair"].mean()

# Effect size (children yes - no)
mean_diff = mean_affairs.get("yes", np.nan) - mean_affairs.get("no", np.nan)
prop_diff = prop_any.get("yes", np.nan) - prop_any.get("no", np.nan)

# Cohen's d on affairs count
cd = cohen_d(df.loc[df["children"] == "yes", "affairs"], df.loc[df["children"] == "no", "affairs"])

# Regression controls (use common Fair model covariates)
# Use formula for OLS on affairs (count), robust SE
covariates = [
    "children_yes",
    "age",
    "yearsmarried",
    "religiousness",
    "education",
    "occupation",
    "rating",
    "C(gender)",
]

# Drop missing
reg_df = df.dropna(subset=["affairs", "children_yes", "age", "yearsmarried", "religiousness", "education", "occupation", "rating", "gender"]).copy()

ols = smf.ols("affairs ~ " + " + ".join(covariates), data=reg_df).fit(cov_type="HC3")

# Logistic regression for any_affair
logit = smf.logit("any_affair ~ " + " + ".join(covariates), data=reg_df).fit(disp=0)

# Marginal effect of children on probability (average marginal effect)
try:
    margeff = logit.get_margeff(at="overall")
    me_children = float(margeff.margeff[0])  # children_yes first
    me_pvalue = float(margeff.pvalues[0])
except Exception:
    me_children = np.nan
    me_pvalue = np.nan

results = {
    "n": len(df),
    "n_yes": int((df["children"] == "yes").sum()),
    "n_no": int((df["children"] == "no").sum()),
    "mean_affairs_yes": float(mean_affairs.get("yes", np.nan)),
    "mean_affairs_no": float(mean_affairs.get("no", np.nan)),
    "mean_diff_yes_minus_no": float(mean_diff),
    "prop_any_yes": float(prop_any.get("yes", np.nan)),
    "prop_any_no": float(prop_any.get("no", np.nan)),
    "prop_diff_yes_minus_no": float(prop_diff),
    "cohen_d_affairs_yes_minus_no": float(cd),
    "ols_children_coef": float(ols.params.get("children_yes", np.nan)),
    "ols_children_p": float(ols.pvalues.get("children_yes", np.nan)),
    "logit_children_coef": float(logit.params.get("children_yes", np.nan)),
    "logit_children_p": float(logit.pvalues.get("children_yes", np.nan)),
    "logit_children_margeff": float(me_children),
    "logit_children_margeff_p": float(me_pvalue),
}

for k, v in results.items():
    print(f"{k}: {v}")
