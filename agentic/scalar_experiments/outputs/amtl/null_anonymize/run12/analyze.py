import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import norm

# Load data
DF_PATH = "amtl.csv"
df = pd.read_csv(DF_PATH)

# Build indicator for human
DF = df.copy()
DF["is_human"] = (DF["feature8"] == "Homo sapiens").astype(int)

# Guard against zero trials
DF = DF[DF["feature4"] > 0].copy()

# Proportion response with weights (binomial)
DF["prop_missing"] = DF["feature3"] / DF["feature4"]

# Fit GLM with binomial family
formula = "prop_missing ~ is_human + feature5 + feature7 + C(feature1)"
model = smf.glm(
    formula=formula,
    data=DF,
    family=sm.families.Binomial(),
    freq_weights=DF["feature4"],
).fit()

# Extract human effect
coef = model.params["is_human"]
se = model.bse["is_human"]
z = coef / se
p = 2 * (1 - norm.cdf(abs(z)))

odds_ratio = float(np.exp(coef))

# Adjusted marginal difference in probability
DF_h = DF.copy()
DF_h["is_human"] = 1
DF_nh = DF.copy()
DF_nh["is_human"] = 0

pred_h = model.predict(DF_h)
pred_nh = model.predict(DF_nh)

avg_diff = float(np.mean(pred_h - pred_nh))

print("coef", coef)
print("se", se)
print("z", z)
print("p", p)
print("odds_ratio", odds_ratio)
print("avg_prob_diff", avg_diff)
print("n", len(DF))
