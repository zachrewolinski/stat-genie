import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
path = "panda_nuts.csv"
df = pd.read_csv(path)

# Ensure categories
for col in ["sex", "help"]:
    df[col] = df[col].astype("category")

# Poisson regression for counts with exposure offset (seconds)
# This models nuts_opened per unit time as a function of age, sex, and help.
model = smf.glm(
    "nuts_opened ~ age + C(sex) + C(help)",
    data=df,
    family=sm.families.Poisson(),
    offset=np.log(df["seconds"])
)
result = model.fit(cov_type="HC0")

print(result.summary())

# Exponentiated coefficients (rate ratios)
params = result.params
conf = result.conf_int()
rate_ratios = np.exp(params)
conf_rr = np.exp(conf)

summary_rr = pd.DataFrame({
    "rate_ratio": rate_ratios,
    "ci_lower": conf_rr[0],
    "ci_upper": conf_rr[1],
    "p_value": result.pvalues,
})
print("\nRate ratios (exp(coef)) with 95% CI:")
print(summary_rr)

# Also compute simple efficiency metric for descriptive context
# Efficiency = nuts_opened per second

df["efficiency"] = df["nuts_opened"] / df["seconds"]
print("\nEfficiency (nuts_opened per second) summary:")
print(df["efficiency"].describe())
