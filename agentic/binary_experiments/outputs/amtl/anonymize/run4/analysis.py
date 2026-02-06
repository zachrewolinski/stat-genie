import pandas as pd
import numpy as np
import statsmodels.api as sm
import patsy

# Load data
path = "amtl.csv"
df = pd.read_csv(path)

# Rename columns for clarity
rename_map = {
    "feature1": "tooth_class",
    "feature2": "specimen_id",
    "feature3": "missing",
    "feature4": "observable",
    "feature5": "age",
    "feature6": "age_unc",
    "feature7": "sex",
    "feature8": "genus",
    "feature9": "region",
}
df = df.rename(columns=rename_map)

# Basic cleaning
for col in ["missing", "observable", "age", "sex"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Keep rows with valid counts and covariates
_df = df.dropna(subset=["missing", "observable", "age", "sex", "tooth_class", "genus"]).copy()
_df = _df[_df["observable"] > 0]
_df["non_missing"] = _df["observable"] - _df["missing"]
_df = _df[_df["non_missing"] >= 0]

# Ensure categorical dtype
_df["tooth_class"] = _df["tooth_class"].astype("category")
_df["genus"] = _df["genus"].astype("category")

# Build design matrix with Homo sapiens as reference
formula = "C(genus, Treatment(reference='Homo sapiens')) + age + sex + C(tooth_class)"
exog = patsy.dmatrix(formula, _df, return_type="dataframe")
design_info = exog.design_info

# Binomial GLM with successes/failures
endog = _df[["missing", "non_missing"]].astype(float)
model = sm.GLM(endog, exog, family=sm.families.Binomial())
result = model.fit()

# Adjusted predicted mean rate per genus (average marginal prediction)
unique_genera = list(_df["genus"].cat.categories)
mean_rates = {}
for g in unique_genera:
    temp = _df.copy()
    temp["genus"] = g
    exog_g = patsy.build_design_matrices([design_info], temp, return_type="dataframe")[0]
    preds = result.predict(exog_g)
    mean_rates[g] = float(np.mean(preds))

# Determine conclusion based on genus coefficients vs Homo sapiens
coef = result.params
pvals = result.pvalues

genus_terms = [c for c in coef.index if c.startswith("C(genus")]
negative_and_sig = True
for term in genus_terms:
    if not (coef[term] < 0 and pvals[term] < 0.05):
        negative_and_sig = False
        break

# Prepare conclusion
if negative_and_sig:
    conclusion = "Yes"
else:
    conclusion = "No"

# Build brief reasoning
# Identify adjusted mean rate ordering
sorted_rates = sorted(mean_rates.items(), key=lambda x: x[1], reverse=True)
rate_str = ", ".join([f"{g}: {mean_rates[g]:.3f}" for g in unique_genera])

reason_lines = []
if conclusion == "Yes":
    reason_lines.append(
        "After adjusting for age, sex, and tooth class, non-human genera show significantly lower AMTL odds than Homo sapiens in the binomial model."
    )
    reason_lines.append(f"Adjusted mean predicted AMTL rates (proportions) by genus are {rate_str}.")
else:
    reason_lines.append(
        "After adjusting for age, sex, and tooth class, at least one non-human genus does not have significantly lower AMTL odds than Homo sapiens in the binomial model."
    )
    reason_lines.append(f"Adjusted mean predicted AMTL rates (proportions) by genus are {rate_str}.")

with open("conclusion.txt", "w", encoding="utf-8") as f:
    f.write(conclusion + "\n")
    f.write(" ".join(reason_lines).strip() + "\n")

# Print a concise summary for inspection
print(result.summary())
print("\nAdjusted mean predicted AMTL rates by genus:")
for g, r in mean_rates.items():
    print(f"{g}: {r:.4f}")
print("\nGenus terms (vs Homo sapiens):")
for term in genus_terms:
    print(f"{term}: coef={coef[term]:.4f}, p={pvals[term]:.4g}")
