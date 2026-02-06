import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from statsmodels.stats.weightstats import ttest_ind
from statsmodels.stats.proportion import proportions_ztest

# Load data
path = "affairs.csv"
df = pd.read_csv(path)

# Basic cleaning / derived variables
df["has_affair"] = (df["affairs"] > 0).astype(int)

# Group summaries
summary = df.groupby("children").agg(
    n=("affairs", "size"),
    mean_affairs=("affairs", "mean"),
    median_affairs=("affairs", "median"),
    prop_any=("has_affair", "mean"),
)

# Mean comparison (t-test) for affairs count
with_children = df.loc[df["children"] == "yes", "affairs"].values
without_children = df.loc[df["children"] == "no", "affairs"].values

t_stat, p_val, _ = ttest_ind(with_children, without_children, usevar="unequal")

# Proportion comparison for any affair
count = np.array([
    df.loc[df["children"] == "yes", "has_affair"].sum(),
    df.loc[df["children"] == "no", "has_affair"].sum(),
])
obs = np.array([
    (df["children"] == "yes").sum(),
    (df["children"] == "no").sum(),
])

z_stat, p_prop = proportions_ztest(count, obs)

# Regression models controlling for covariates
# OLS on affairs count (robust SEs)
ols_model = smf.ols(
    "affairs ~ C(children) + C(gender) + age + yearsmarried + religiousness + education + occupation + rating",
    data=df,
).fit(cov_type="HC3")

# Logit on any affair
logit_model = smf.logit(
    "has_affair ~ C(children) + C(gender) + age + yearsmarried + religiousness + education + occupation + rating",
    data=df,
).fit(disp=False)

# Extract key coefficients
ols_coef = ols_model.params.get("C(children)[T.yes]", np.nan)
ols_p = ols_model.pvalues.get("C(children)[T.yes]", np.nan)

logit_coef = logit_model.params.get("C(children)[T.yes]", np.nan)
logit_p = logit_model.pvalues.get("C(children)[T.yes]", np.nan)

# Average marginal effect of children on probability of any affair
# Approximate by predicting with children yes/no and averaging
pred_yes = logit_model.predict(df.assign(children="yes"))
pred_no = logit_model.predict(df.assign(children="no"))

avg_marginal = (pred_yes - pred_no).mean()

# Output results
print("Group summary by children:\n", summary, "\n")
print("T-test (affairs count) children yes vs no: t=%.3f, p=%.4f" % (t_stat, p_val))
print("Proportion test (any affair) children yes vs no: z=%.3f, p=%.4f" % (z_stat, p_prop))
print("\nOLS coefficient for children=yes (affairs count): %.3f, p=%.4f" % (ols_coef, ols_p))
print("Logit coefficient for children=yes (any affair): %.3f, p=%.4f" % (logit_coef, logit_p))
print("Average marginal effect on P(any affair) for children=yes: %.3f" % avg_marginal)

# Save a compact results table for reference
results = {
    "mean_affairs_yes": summary.loc["yes", "mean_affairs"],
    "mean_affairs_no": summary.loc["no", "mean_affairs"],
    "prop_any_yes": summary.loc["yes", "prop_any"],
    "prop_any_no": summary.loc["no", "prop_any"],
    "t_stat": t_stat,
    "t_p": p_val,
    "z_stat": z_stat,
    "z_p": p_prop,
    "ols_coef_children_yes": ols_coef,
    "ols_p_children_yes": ols_p,
    "logit_coef_children_yes": logit_coef,
    "logit_p_children_yes": logit_p,
    "avg_marginal_effect": avg_marginal,
}

pd.DataFrame([results]).to_csv("analysis_results.csv", index=False)
