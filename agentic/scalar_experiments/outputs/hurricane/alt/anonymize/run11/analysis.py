import json
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import spearmanr

# Load data
path = "hurricane.csv"
df = pd.read_csv(path)

# Outcome: fatalities (log1p for skew)
df["log_deaths"] = np.log1p(df["feature8"])

# Models focusing on name femininity vs deaths (proxy for precaution effectiveness)
# feature4: coder-rated femininity index
# feature12: MTurk-rated femininity index
# feature6: binary female (1) vs male (0)

m1 = smf.ols("log_deaths ~ feature4", data=df).fit(cov_type="HC3")
m2 = smf.ols("log_deaths ~ feature4 + feature7 + feature5 + feature13", data=df).fit(cov_type="HC3")
m3 = smf.ols(
    "log_deaths ~ feature4 + feature7 + feature5 + feature13 + np.log1p(feature14)",
    data=df,
).fit(cov_type="HC3")
m4 = smf.ols("log_deaths ~ feature12 + feature7 + feature5 + feature13", data=df).fit(
    cov_type="HC3"
)
m5 = smf.ols("log_deaths ~ feature6 + feature7 + feature5 + feature13", data=df).fit(
    cov_type="HC3"
)

rho4, p_rho4 = spearmanr(df["feature4"], df["feature8"])
rho12, p_rho12 = spearmanr(df["feature12"], df["feature8"])

n = len(df)

# Extract key stats
coef_m2 = m2.params["feature4"]
p_m2 = m2.pvalues["feature4"]
coef_m3 = m3.params["feature4"]
p_m3 = m3.pvalues["feature4"]
coef_m4 = m4.params["feature12"]
p_m4 = m4.pvalues["feature12"]
coef_m5 = m5.params["feature6"]
p_m5 = m5.pvalues["feature6"]
coef_m1 = m1.params["feature4"]
p_m1 = m1.pvalues["feature4"]

# Decide response: evidence does not support the claim
response = 30  # No, with low-to-moderate confidence given small, non-significant effects

explanation = (
    f"Analyzed {n} U.S. landfalling hurricanes (1950–2012). The claim implies that more feminine names "
    f"lead to fewer precautions, which should show up as higher fatalities after accounting for storm severity. "
    f"Simple association is near zero (Spearman rho for femininity index vs deaths = {rho4:.3f}, p={p_rho4:.3f}; "
    f"MTurk rating rho = {rho12:.3f}, p={p_rho12:.3f}). In OLS with robust SEs, femininity index is not a "
    f"significant predictor of log deaths: b={coef_m1:.3f}, p={p_m1:.3f} unadjusted; b={coef_m2:.3f}, p={p_m2:.3f} "
    f"controlling for category, minimum pressure, and wind; and b={coef_m3:.3f}, p={p_m3:.3f} when also controlling for "
    f"logged damage. Alternative femininity rating (feature12) is also non‑significant (b={coef_m4:.3f}, p={p_m4:.3f}), "
    f"and the binary female-name indicator is non‑significant (b={coef_m5:.3f}, p={p_m5:.3f}). Overall, the data do "
    f"not show a statistically reliable relationship between name femininity and fatalities (a proxy for precautionary "
    f"behavior), so the evidence does not support the research claim." 
)

with open("conclusion.txt", "w") as f:
    json.dump({"response": int(response), "explanation": explanation}, f)
