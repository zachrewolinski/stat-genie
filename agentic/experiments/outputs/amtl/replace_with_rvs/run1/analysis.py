import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
path = "amtl.csv"
df = pd.read_csv(path)

# Basic cleaning
# Ensure numeric columns
for col in ["num_amtl", "sockets", "age", "prob_male"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Drop rows with missing essential fields
req = ["num_amtl", "sockets", "age", "prob_male", "tooth_class", "genus"]
df = df.dropna(subset=req).copy()

# Filter impossible values
df = df[(df["sockets"] > 0) & (df["num_amtl"] >= 0) & (df["num_amtl"] <= df["sockets"])].copy()

# Indicator for modern humans
# The genus column includes "Homo sapiens"
df["is_human"] = (df["genus"] == "Homo sapiens").astype(int)

# Standardize age to improve model stability
age_mean = df["age"].mean()
age_std = df["age"].std(ddof=0)
if age_std == 0:
    df["age_z"] = 0.0
else:
    df["age_z"] = (df["age"] - age_mean) / age_std

# Prepare response for binomial GLM (successes, failures)
endog = np.column_stack([df["num_amtl"].values, (df["sockets"] - df["num_amtl"]).values])

# Model formula: AMTL rate ~ human indicator + age + sex prob + tooth class
# Use tooth_class as categorical
exog = sm.add_constant(pd.get_dummies(df[["is_human", "age_z", "prob_male", "tooth_class"]],
                                      columns=["tooth_class"], drop_first=True))

model = sm.GLM(endog, exog, family=sm.families.Binomial())
result = model.fit()

# Extract human effect
coef = result.params.get("is_human", np.nan)
se = result.bse.get("is_human", np.nan)
pval = result.pvalues.get("is_human", np.nan)

# Save a brief text report
with open("analysis.txt", "w") as f:
    f.write("Binomial GLM for AMTL rate with predictors: is_human, age_z, prob_male, tooth_class\n")
    f.write(result.summary().as_text())
    f.write("\n\nHuman effect (is_human):\n")
    f.write(f"coef={coef:.4f}, se={se:.4f}, p={pval:.6f}\n")
    f.write(f"Odds ratio={np.exp(coef):.4f}\n")

# Determine conclusion based on coefficient sign and significance
alpha = 0.05
higher = (coef > 0) and (pval < alpha)

with open("conclusion.txt", "w") as f:
    f.write("Yes\n" if higher else "No\n")
    if higher:
        f.write("After accounting for age, sex probability, and tooth class, the human indicator has a positive and statistically significant effect on AMTL rate (logit model).\n")
    else:
        f.write("After accounting for age, sex probability, and tooth class, the human indicator is not positive and statistically significant for AMTL rate in the binomial model.\n")
