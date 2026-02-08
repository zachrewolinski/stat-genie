import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

# Load data
_df = pd.read_csv("affairs.csv")

# Basic preprocessing
_df = _df.copy()
_df["children_yes"] = (_df["children"].astype(str).str.lower() == "yes").astype(int)

# Descriptives
by_child = _df.groupby("children")["affairs"].agg(["mean","median","count","std"])

# Difference in means test (Welch t-test)
with_children = _df.loc[_df["children_yes"] == 1, "affairs"].to_numpy()
without_children = _df.loc[_df["children_yes"] == 0, "affairs"].to_numpy()

t_stat, p_val = stats.ttest_ind(with_children, without_children, equal_var=False)

# Proportion with any affairs
_df["any_affair"] = (_df["affairs"] > 0).astype(int)
prop_by_child = _df.groupby("children")["any_affair"].mean()

# Logistic regression for any affair
logit_model = smf.logit(
    "any_affair ~ children_yes + age + yearsmarried + C(gender) + religiousness + education + occupation + rating",
    data=_df,
).fit(disp=False)

# Poisson regression for count of affairs (robust SEs)
poisson_model = smf.poisson(
    "affairs ~ children_yes + age + yearsmarried + C(gender) + religiousness + education + occupation + rating",
    data=_df,
).fit(disp=False)

# Extract key results
logit_coef = logit_model.params["children_yes"]
logit_p = logit_model.pvalues["children_yes"]
poisson_coef = poisson_model.params["children_yes"]
poisson_p = poisson_model.pvalues["children_yes"]

# Convert to odds ratio and incidence rate ratio for interpretation
logit_or = float(np.exp(logit_coef))
poisson_irr = float(np.exp(poisson_coef))

# Print summary for manual inspection
print("Descriptives (affairs by children):\n", by_child, "\n")
print("Welch t-test: t=%.3f, p=%.4g" % (t_stat, p_val))
print("Proportion any affair by children:\n", prop_by_child, "\n")
print("Logit children_yes coef=%.4f p=%.4g OR=%.3f" % (logit_coef, logit_p, logit_or))
print("Poisson children_yes coef=%.4f p=%.4g IRR=%.3f" % (poisson_coef, poisson_p, poisson_irr))
