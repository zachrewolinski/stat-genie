import pandas as pd
import statsmodels.api as sm

# Load data
DATA_PATH = "amtl.csv"
df = pd.read_csv(DATA_PATH)

# Basic cleaning
# Keep rows with valid counts
counts_ok = df["sockets"].notna() & df["num_amtl"].notna()
df = df[counts_ok].copy()
df = df[df["sockets"] >= df["num_amtl"]].copy()

# Create human indicator
df_is_human = df["genus"].astype(str).str.strip().eq("Homo sapiens")
df["is_human"] = df_is_human.astype(int)

# Prepare design matrix
# Include age, sex probability, tooth class (categorical), and human indicator
exog = pd.get_dummies(
    df[["is_human", "age", "prob_male", "tooth_class"]],
    columns=["tooth_class"],
    drop_first=True,
)
exog = sm.add_constant(exog, has_constant="add")

# Binomial response as counts
endog = df[["num_amtl", "sockets"]].copy()
endog["non_amtl"] = endog["sockets"] - endog["num_amtl"]
endog = endog[["num_amtl", "non_amtl"]].values

# Fit binomial GLM
model = sm.GLM(endog, exog, family=sm.families.Binomial())
result = model.fit()

# Save a brief summary to a text file for inspection if needed
with open("analysis_summary.txt", "w") as f:
    f.write(result.summary().as_text())

# Extract key result for is_human
coef = result.params.get("is_human", float("nan"))
se = result.bse.get("is_human", float("nan"))
pval = result.pvalues.get("is_human", float("nan"))

# Compute odds ratio
import numpy as np
odds_ratio = float(np.exp(coef))

print("is_human coef:", coef)
print("is_human SE:", se)
print("is_human p-value:", pval)
print("is_human odds ratio:", odds_ratio)
