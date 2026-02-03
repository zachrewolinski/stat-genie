import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
_df = pd.read_csv("amtl.csv")

# Keep rows that are valid for a binomial model (successes <= trials)
# This avoids invalid proportions (>1) when num_amtl exceeds sockets.
_df = _df[_df["num_amtl"] <= _df["sockets"]].copy()

# Binary indicator for modern humans vs non-human primates
_df["human"] = (_df["genus"] == "Homo sapiens").astype(int)

# Binomial response: successes (AMTL) and failures (non-AMTL sockets)
_df["failures"] = _df["sockets"] - _df["num_amtl"]

# Fit binomial regression, adjusting for age, sex estimate, and tooth class
model = smf.glm(
    "num_amtl + failures ~ human + age + prob_male + C(tooth_class)",
    data=_df,
    family=sm.families.Binomial(),
)

# Cluster-robust SE by specimen (repeated measures per specimen)
result = model.fit(cov_type="cluster", cov_kwds={"groups": _df["specimen"]})

# Print key results for inspection
print(result.summary())

coef = result.params["human"]
se = result.bse["human"]
p = result.pvalues["human"]

or_human = float(np.exp(coef))
ci_low = float(np.exp(coef - 1.96 * se))
ci_high = float(np.exp(coef + 1.96 * se))

print("\nHuman effect (odds ratio):", or_human)
print("95% CI:", (ci_low, ci_high))
print("p-value:", p)
