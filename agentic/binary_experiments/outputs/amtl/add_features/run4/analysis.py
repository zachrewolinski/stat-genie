import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
path = "amtl.csv"
df = pd.read_csv(path)

# Keep relevant columns
needed = ["num_amtl", "sockets", "age", "prob_male", "tooth_class", "genus"]
missing_cols = [c for c in needed if c not in df.columns]
if missing_cols:
    raise ValueError(f"Missing columns: {missing_cols}")

sub = df[needed].copy()

# Basic cleaning
sub = sub.dropna(subset=["num_amtl", "sockets", "age", "prob_male", "tooth_class", "genus"])
sub = sub[(sub["sockets"] > 0) & (sub["num_amtl"] >= 0) & (sub["num_amtl"] <= sub["sockets"])]

# Create human indicator
sub["is_human"] = (sub["genus"] == "Homo sapiens").astype(int)

# Response as proportion with binomial weights
sub["amtl_rate"] = sub["num_amtl"] / sub["sockets"]

# Fit binomial GLM
model = smf.glm(
    formula="amtl_rate ~ is_human + age + prob_male + C(tooth_class)",
    data=sub,
    family=sm.families.Binomial(),
    freq_weights=sub["sockets"],
)
result = model.fit()

# Extract effect for human indicator
coef = result.params.get("is_human", np.nan)
se = result.bse.get("is_human", np.nan)
pval = result.pvalues.get("is_human", np.nan)

# Odds ratio and 95% CI
or_est = float(np.exp(coef))
ci_low = float(np.exp(coef - 1.96 * se))
ci_high = float(np.exp(coef + 1.96 * se))

# Save a short text report
with open("analysis_results.txt", "w") as f:
    f.write(result.summary().as_text())
    f.write("\n\n")
    f.write(f"is_human coef: {coef:.4f}\n")
    f.write(f"is_human p-value: {pval:.6g}\n")
    f.write(f"Odds ratio: {or_est:.4f} (95% CI {ci_low:.4f}, {ci_high:.4f})\n")
    f.write(f"N used: {len(sub)}\n")

print("Rows used:", len(sub))
print("is_human coef:", coef)
print("p-value:", pval)
print("OR (95% CI):", or_est, (ci_low, ci_high))
