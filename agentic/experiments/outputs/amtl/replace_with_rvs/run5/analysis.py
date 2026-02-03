import numpy as np
import pandas as pd
import statsmodels.api as sm
import patsy

# Load data
csv_path = "amtl.csv"
df = pd.read_csv(csv_path)

# Basic cleaning
needed_cols = ["num_amtl", "sockets", "age", "prob_male", "tooth_class", "genus"]
df = df.dropna(subset=needed_cols).copy()

# Ensure valid sockets and counts
mask = (df["sockets"] > 0) & (df["num_amtl"] >= 0) & (df["num_amtl"] <= df["sockets"])
df = df.loc[mask].copy()

# Binary indicator for modern humans vs non-human primates
# Treat "Homo sapiens" as modern humans; all others as non-human primates
df["is_human"] = (df["genus"] == "Homo sapiens").astype(int)

# Build design matrices
formula = "is_human + age + prob_male + C(tooth_class)"
X = patsy.dmatrix(formula, df, return_type="dataframe")

# Endog as successes/failures for binomial GLM
successes = df["num_amtl"].astype(float).to_numpy()
failures = (df["sockets"] - df["num_amtl"]).astype(float).to_numpy()
Y = np.column_stack([successes, failures])

model = sm.GLM(Y, X, family=sm.families.Binomial())
result = model.fit()

# Extract coefficient and inference for is_human
coef = result.params.get("is_human", float("nan"))
se = result.bse.get("is_human", float("nan"))
pval = result.pvalues.get("is_human", float("nan"))

# Odds ratio and 95% CI
or_val = float("nan")
ci_low = float("nan")
ci_high = float("nan")
if np.isfinite(coef) and np.isfinite(se):
    or_val = float(np.exp(coef))
    ci_low = float(np.exp(coef - 1.96 * se))
    ci_high = float(np.exp(coef + 1.96 * se))

# Save a concise report
with open("analysis_summary.txt", "w") as f:
    f.write("Binomial GLM: num_amtl/sockets ~ is_human + age + prob_male + C(tooth_class)\n")
    f.write(result.summary().as_text())
    f.write("\n\n")
    f.write(f"is_human coef: {coef:.6f}\n")
    f.write(f"is_human p-value: {pval:.6g}\n")
    f.write(f"is_human odds ratio: {or_val:.6f}\n")
    f.write(f"is_human OR 95% CI: [{ci_low:.6f}, {ci_high:.6f}]\n")

print(result.summary())
print("\nKey effect (is_human):")
print(f"coef={coef:.6f}, p={pval:.6g}, OR={or_val:.6f}, CI=[{ci_low:.6f}, {ci_high:.6f}]")
