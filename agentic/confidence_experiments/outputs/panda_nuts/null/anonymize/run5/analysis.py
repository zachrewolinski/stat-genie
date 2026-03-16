import json
import pandas as pd
import statsmodels.formula.api as smf

# Load data
csv_path = "panda_nuts.csv"
df = pd.read_csv(csv_path)

# Rename for clarity
# feature2: age, feature3: sex, feature5: nuts opened, feature6: duration, feature7: help

df = df.copy()

# Compute efficiency: nuts per second
# Avoid division by zero (feature6 min is 2.5 per metadata)
df["efficiency"] = df["feature5"] / df["feature6"]

# Normalize categorical values
# feature3 has 'f'/'m'; feature7 has 'y'/'N' per metadata
# Make consistent case
for col in ["feature3", "feature7"]:
    df[col] = df[col].astype(str).str.strip()

# Map help to yes/no labels for readability
help_map = {"y": "yes", "Y": "yes", "N": "no", "n": "no"}
df["help"] = df["feature7"].map(help_map)

# Keep original sex values
sex_map = {"f": "f", "F": "f", "m": "m", "M": "m"}
df["sex"] = df["feature3"].map(sex_map)

# Drop rows with missing key values
analysis_df = df[["efficiency", "feature2", "sex", "help"]].dropna()
analysis_df = analysis_df.rename(columns={"feature2": "age"})

# Fit OLS with robust SEs
model = smf.ols("efficiency ~ age + C(sex) + C(help)", data=analysis_df).fit(cov_type="HC3")

# Collect results
params = model.params.to_dict()
pvalues = model.pvalues.to_dict()

# Overall model F-test
f_pvalue = model.f_pvalue

# Prepare summary stats for explanation
result = {
    "n": int(analysis_df.shape[0]),
    "params": params,
    "pvalues": pvalues,
    "f_pvalue": float(f_pvalue) if f_pvalue is not None else None,
    "r2": float(model.rsquared),
}

print(json.dumps(result, indent=2))
