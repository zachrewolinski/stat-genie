import pandas as pd
import numpy as np
import statsmodels.api as sm

# Load data
csv_path = "amtl.csv"
info_path = "info.json"

_df = pd.read_csv(csv_path)

# Map columns
col_tooth_class = "feature1"
col_missing = "feature3"
col_sockets = "feature4"
col_age = "feature5"
col_sex = "feature7"
col_genus = "feature8"

# Basic cleaning
_df = _df.copy()
_df[col_missing] = pd.to_numeric(_df[col_missing], errors="coerce")
_df[col_sockets] = pd.to_numeric(_df[col_sockets], errors="coerce")
_df[col_age] = pd.to_numeric(_df[col_age], errors="coerce")
_df[col_sex] = pd.to_numeric(_df[col_sex], errors="coerce")

# Valid rows: non-missing, sockets>0, missing between 0 and sockets
_df = _df.dropna(subset=[col_missing, col_sockets, col_age, col_sex, col_genus, col_tooth_class])
_df = _df[_df[col_sockets] > 0]
_df = _df[(
    (_df[col_missing] >= 0) &
    (_df[col_missing] <= _df[col_sockets])
)]

# Binary human indicator
_df["is_human"] = (_df[col_genus] == "Homo sapiens").astype(int)

# Create design matrix with categorical tooth class
X = pd.get_dummies(_df[["is_human", col_age, col_sex, col_tooth_class]], columns=[col_tooth_class], drop_first=True)
X = sm.add_constant(X, has_constant="add")

# Endog as successes/failures
missing = _df[col_missing].astype(int)
present = (_df[col_sockets] - _df[col_missing]).astype(int)
endog = np.column_stack([missing, present])

model = sm.GLM(endog, X, family=sm.families.Binomial())
res = model.fit()

coef = res.params.get("is_human", np.nan)
se = res.bse.get("is_human", np.nan)
pval = res.pvalues.get("is_human", np.nan)

# Compute z-score
z = coef / se if pd.notna(coef) and pd.notna(se) and se != 0 else np.nan

# Convert to Likert score (-100 to 100)
if pd.isna(coef) or pd.isna(z):
    score = 0
else:
    sign = 1 if coef > 0 else -1 if coef < 0 else 0
    # Strength mapping: |z| >= 3 => 100, |z| <= 0.5 => ~17
    strength = int(round(min(100, (abs(z) / 3.0) * 100)))
    # If not statistically suggestive, dampen
    if pd.notna(pval) and pval > 0.1:
        strength = min(strength, 20)
    score = int(sign * strength)

# Write conclusion
with open("conclusion.txt", "w", encoding="utf-8") as f:
    f.write(str(int(score)))

# Also write a short report for debugging (not required but useful)
with open("analysis_report.txt", "w", encoding="utf-8") as f:
    f.write("Rows used: %d\n" % len(_df))
    f.write("Coef is_human: %s\n" % coef)
    f.write("SE is_human: %s\n" % se)
    f.write("z: %s\n" % z)
    f.write("pval: %s\n" % pval)
    f.write("Score: %d\n" % score)
