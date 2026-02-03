import pandas as pd
import numpy as np
import statsmodels.api as sm
import patsy

# Load data
csv_path = "/home/chenwang/stat-genie/agentic/experiments/outputs/amtl/replace_with_rvs/run3/amtl.csv"
df = pd.read_csv(csv_path)

# Basic cleaning: keep rows with required fields
required_cols = ["num_amtl", "sockets", "age", "prob_male", "tooth_class", "genus"]
df = df.dropna(subset=required_cols).copy()

# Ensure numeric types
for col in ["num_amtl", "sockets", "age", "prob_male"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=["num_amtl", "sockets", "age", "prob_male"])

# Filter to valid counts
# sockets must be positive and num_amtl between 0 and sockets
mask = (df["sockets"] > 0) & (df["num_amtl"] >= 0) & (df["num_amtl"] <= df["sockets"])
df = df.loc[mask].copy()

# Create human indicator
# Homo sapiens in dataset is labeled "Homo sapiens"
df["is_human"] = (df["genus"] == "Homo sapiens").astype(int)

# Fit binomial GLM with logit link using successes/failures
# Controls: age, sex probability, tooth class
predictor_formula = "is_human + age + prob_male + C(tooth_class)"
y, X = patsy.dmatrices(f"num_amtl ~ {predictor_formula}", data=df, return_type="dataframe")
endog = np.column_stack([df["num_amtl"].values, (df["sockets"] - df["num_amtl"]).values])
model = sm.GLM(endog, X, family=sm.families.Binomial())
result = model.fit()

# Extract human effect
coef = result.params.get("is_human", np.nan)
se = result.bse.get("is_human", np.nan)
pval = result.pvalues.get("is_human", np.nan)

# Odds ratio and 95% CI
or_est = np.exp(coef)
ci_low = np.exp(coef - 1.96 * se)
ci_high = np.exp(coef + 1.96 * se)

print("Rows used:", len(df))
print(result.summary())
print("\nHuman effect (is_human):")
print(f"  log-odds coef = {coef:.4f}")
print(f"  OR = {or_est:.4f}")
print(f"  95% CI OR = [{ci_low:.4f}, {ci_high:.4f}]")
print(f"  p-value = {pval:.6f}")
