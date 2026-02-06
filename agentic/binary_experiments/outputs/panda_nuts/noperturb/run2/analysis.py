import pandas as pd
import statsmodels.formula.api as smf

# Load data
DATA_PATH = "panda_nuts.csv"

df = pd.read_csv(DATA_PATH)

# Clean / prepare
# Standardize categorical labels
for col in ["sex", "help", "hammer"]:
    if col in df.columns:
        df[col] = df[col].astype(str).str.strip()

# Efficiency: nuts opened per second
# Avoid divide-by-zero (none expected since min seconds 2.5)

df["efficiency"] = df["nuts_opened"] / df["seconds"]

# Basic model: efficiency ~ age + sex + help
# Treat sex and help as categorical
model = smf.ols("efficiency ~ age + C(sex) + C(help)", data=df).fit()

# Extract p-values
pvals = model.pvalues

# Determine influence at alpha=0.05
alpha = 0.05
influential = {
    "age": pvals.get("age", 1.0) < alpha,
    "sex": any(name.startswith("C(sex)") and p < alpha for name, p in pvals.items()),
    "help": any(name.startswith("C(help)") and p < alpha for name, p in pvals.items()),
}

# Save results for inspection
summary_path = "analysis_summary.txt"
with open(summary_path, "w") as f:
    f.write(model.summary().as_text())
    f.write("\n\nP-values:\n")
    for k, v in pvals.items():
        f.write(f"{k}: {v}\n")

# Construct conclusion
# If any of the three factors is significant, answer Yes
answer = "Yes" if any(influential.values()) else "No"

# Brief reasoning based on which variables significant
sig_vars = [k for k, v in influential.items() if v]
if sig_vars:
    reason = (
        "Regression on nuts-opened-per-second shows statistically significant effects for "
        + ", ".join(sig_vars)
        + " (p < 0.05), indicating these factors influence efficiency."
    )
else:
    reason = (
        "Regression on nuts-opened-per-second finds no statistically significant effects "
        "for age, sex, or help (all p >= 0.05), so there is no evidence these factors influence efficiency."
    )

with open("conclusion.txt", "w") as f:
    f.write(answer + "\n")
    f.write(reason + "\n")

print("Model fit complete. Conclusion written to conclusion.txt.")
