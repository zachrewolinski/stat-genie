import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.proportion import proportions_ztest

# Load data
path = "affairs.csv"
df = pd.read_csv(path)

# Rename columns for clarity
col_map = {
    "feature1": "id",
    "feature2": "affairs",
    "feature3": "gender",
    "feature4": "age",
    "feature5": "years_married",
    "feature6": "children",
    "feature7": "religiosity",
    "feature8": "education",
    "feature9": "occupation",
    "feature10": "marriage_rating",
}
df = df.rename(columns=col_map)

# Derived variables

df["children_yes"] = (df["children"].str.lower() == "yes").astype(int)
df["any_affair"] = (df["affairs"] > 0).astype(int)
df["log1p_affairs"] = np.log1p(df["affairs"])

# Group stats

grp = df.groupby("children_yes")
summary = grp["affairs"].agg(["count", "mean", "median"])
summary["prop_any_affair"] = grp["any_affair"].mean()

# Two-proportion z-test for any affair
counts = grp["any_affair"].sum()
ns = grp["any_affair"].count()
z_stat, p_value = proportions_ztest(counts, ns)

# Difference in mean affairs
mean_no_children = summary.loc[0, "mean"]
mean_children = summary.loc[1, "mean"]

# Logistic regression: any affair
logit_model = smf.logit(
    "any_affair ~ children_yes + C(gender) + age + years_married + religiosity + education + occupation + marriage_rating",
    data=df,
).fit(disp=False)

# OLS on log1p(affairs) with robust SE
ols_model = smf.ols(
    "log1p_affairs ~ children_yes + C(gender) + age + years_married + religiosity + education + occupation + marriage_rating",
    data=df,
).fit(cov_type="HC3")

# Extract key coefficients
logit_coef = logit_model.params["children_yes"]
logit_p = logit_model.pvalues["children_yes"]
logit_or = np.exp(logit_coef)

ols_coef = ols_model.params["children_yes"]
ols_p = ols_model.pvalues["children_yes"]

print("Group summary (by children_yes: 0=no, 1=yes):")
print(summary)
print()
print(f"Two-proportion z-test for any affair: z={z_stat:.3f}, p={p_value:.4f}")
print(f"Mean affairs (no children): {mean_no_children:.3f}")
print(f"Mean affairs (children): {mean_children:.3f}")
print(f"Mean difference (children - no children): {mean_children - mean_no_children:.3f}")
print()
print("Logit model (any affair) key effect:")
print(f"children_yes coef={logit_coef:.3f}, odds ratio={logit_or:.3f}, p={logit_p:.4f}")
print()
print("OLS on log1p(affairs) key effect:")
print(f"children_yes coef={ols_coef:.3f}, p={ols_p:.4f}")
