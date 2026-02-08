import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm

# Load data

df = pd.read_csv("amtl.csv")

# Create human indicator

df["human"] = (df["genus"] == "Homo sapiens").astype(int)

# Ensure counts are valid

df = df.copy()

df["failures"] = df["sockets"] - df["num_amtl"]

# Filter any invalid rows just in case

df = df[(df["num_amtl"] >= 0) & (df["failures"] >= 0) & (df["sockets"] > 0)]

# Fit binomial GLM with successes/failures

model = smf.glm(
    "num_amtl + failures ~ human + age + prob_male + C(tooth_class)",
    data=df,
    family=sm.families.Binomial(),
)

result = model.fit()

coef = result.params["human"]
se = result.bse["human"]
z = coef / se if se != 0 else 0.0
pvalue = result.pvalues["human"]

# Convert evidence to Likert scale [-100, 100]
# Positive coef => higher AMTL in humans (Yes). Negative => lower (No).
# Scale by z-score to reflect strength; tanh keeps within bounds.
score = int(np.round(100 * np.tanh(z / 3)))

# Save conclusion
with open("conclusion.txt", "w", encoding="utf-8") as f:
    f.write(str(score))

# Print a brief summary for inspection
print("human coef:", coef)
print("human se:", se)
print("human z:", z)
print("human pvalue:", pvalue)
print("likert score:", score)
