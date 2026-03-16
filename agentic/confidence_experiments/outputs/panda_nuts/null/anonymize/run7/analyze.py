import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api as sm
from statsmodels.stats.anova import anova_lm
from scipy import stats

# Load data
path = "panda_nuts.csv"
df = pd.read_csv(path)

# Rename columns for clarity
rename_map = {
    "feature1": "id",
    "feature2": "age",
    "feature3": "sex",
    "feature4": "hammer",
    "feature5": "nuts_opened",
    "feature6": "duration_sec",
    "feature7": "help",
}
df = df.rename(columns=rename_map)

# Basic cleaning
# Ensure categorical types
for col in ["sex", "hammer", "help"]:
    df[col] = df[col].astype(str)

# Compute efficiency: nuts per minute
# Avoid divide by zero if any duration is zero
if (df["duration_sec"] <= 0).any():
    df = df[df["duration_sec"] > 0].copy()

df["efficiency_npm"] = df["nuts_opened"] / df["duration_sec"] * 60.0

# OLS model
ols_model = smf.ols("efficiency_npm ~ age + C(sex) + C(help)", data=df).fit()

# ANOVA for overall effect of each predictor
anova = anova_lm(ols_model, typ=2)

# Mixed effects with random intercept by individual, if possible
mixed_result = None
try:
    mixed_model = smf.mixedlm("efficiency_npm ~ age + C(sex) + C(help)", data=df, groups=df["id"])
    mixed_result = mixed_model.fit(reml=False, method="lbfgs", maxiter=200)
except Exception as e:
    mixed_result = None

# GLM for nuts opened with log(duration) offset (rate model)
glm_poisson = smf.glm(
    "nuts_opened ~ age + C(sex) + C(help)",
    data=df,
    family=sm.families.Poisson(),
    offset=np.log(df["duration_sec"]),
).fit()

glm_nb = smf.glm(
    "nuts_opened ~ age + C(sex) + C(help)",
    data=df,
    family=sm.families.NegativeBinomial(),
    offset=np.log(df["duration_sec"]),
).fit()

print("N rows:", len(df))
print("Unique individuals:", df["id"].nunique())
print("Efficiency summary (nuts/min):")
print(df["efficiency_npm"].describe())

print("\nOLS summary:")
print(ols_model.summary())
print("\nANOVA (Type II):")
print(anova)

if mixed_result is not None:
    print("\nMixedLM summary:")
    print(mixed_result.summary())
else:
    print("\nMixedLM not available / failed to converge")

print("\nGLM Poisson (with log-duration offset):")
print(glm_poisson.summary())

print("\nGLM Negative Binomial (with log-duration offset):")
print(glm_nb.summary())

# Nonparametric checks
age_spearman = stats.spearmanr(df["age"], df["efficiency_npm"], nan_policy="omit")

sex_groups = [df.loc[df["sex"] == s, "efficiency_npm"] for s in df["sex"].unique()]
help_groups = [df.loc[df["help"] == h, "efficiency_npm"] for h in df["help"].unique()]

sex_test = stats.mannwhitneyu(sex_groups[0], sex_groups[1], alternative="two-sided")
help_test = stats.mannwhitneyu(help_groups[0], help_groups[1], alternative="two-sided")

print("\nNonparametric checks:")
print("Spearman age vs efficiency:", age_spearman)
print("Mann-Whitney sex groups:", sex_test)
print("Mann-Whitney help groups:", help_test)
