import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
path = "amtl.csv"
df = pd.read_csv(path)

# Basic cleaning: drop rows with missing key fields
key_cols = ["num_amtl", "sockets", "age", "prob_male", "tooth_class", "genus"]
df = df.dropna(subset=key_cols).copy()

# Ensure numeric types
for col in ["num_amtl", "sockets", "age", "prob_male"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Drop any rows with nonpositive sockets or invalid counts
mask_valid = (df["sockets"] > 0) & (df["num_amtl"] >= 0) & (df["num_amtl"] <= df["sockets"])
df = df.loc[mask_valid].copy()

# Categorical variables
for col in ["tooth_class", "genus"]:
    df[col] = df[col].astype("category")

# Relevel genus so Homo sapiens is reference
if "Homo sapiens" in df["genus"].cat.categories:
    df["genus"] = df["genus"].cat.reorder_categories(
        ["Homo sapiens"] + [c for c in df["genus"].cat.categories if c != "Homo sapiens"],
        ordered=False,
    )

# Fit binomial GLM with counts
formula = "num_amtl + I(sockets - num_amtl) ~ age + prob_male + C(tooth_class) + C(genus)"
result = smf.glm(formula=formula, data=df, family=sm.families.Binomial()).fit()

# Compute adjusted predicted AMTL rate by genus by averaging predictions over observed covariates
# Keep age, prob_male, tooth_class as observed; set genus to each value
adj_rates = {}
for genus in df["genus"].cat.categories:
    df_tmp = df.copy()
    df_tmp["genus"] = genus
    pred = result.predict(df_tmp)
    adj_rates[genus] = pred.mean()

# Extract coefficients and p-values for genus effects
coef_table = result.summary2().tables[1]

print("N rows used:", len(df))
print("Adjusted mean AMTL rate by genus (model-based):")
for g, r in adj_rates.items():
    print(f"  {g}: {r:.4f}")

print("\nGenus coefficients (relative to Homo sapiens):")
for idx in coef_table.index:
    if "C(genus)" in idx:
        coef = coef_table.loc[idx, "Coef."]
        pval = coef_table.loc[idx, "P>|z|"]
        print(f"  {idx}: coef={coef:.4f}, p={pval:.4g}")
