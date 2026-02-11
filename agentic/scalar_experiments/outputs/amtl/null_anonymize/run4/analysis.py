import json
import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats

# Load data
csv_path = "amtl.csv"
df = pd.read_csv(csv_path)

# Rename for clarity
# feature1: tooth class
# feature3: missing teeth count
# feature4: observable sockets
# feature5: age
# feature7: sex estimate
# feature8: genus

# Build response as successes/failures
missing = df["feature3"].astype(float)
observed = df["feature4"].astype(float)
# Guard against any zeros or invalid rows
valid = (observed > 0) & (missing >= 0) & (missing <= observed)

df = df.loc[valid].copy()
missing = df["feature3"].astype(float)
observed = df["feature4"].astype(float)

# Human indicator
human_label = "Homo sapiens"
df["is_human"] = (df["feature8"] == human_label).astype(int)

# Build 2-col endog for binomial glm
endog = np.column_stack([missing, observed - missing])

# Build model
# Control for age, sex, tooth class
formula = "is_human + feature5 + feature7 + C(feature1)"

model = sm.GLM(endog, sm.add_constant(
    pd.get_dummies(df[["is_human", "feature5", "feature7", "feature1"]], columns=["feature1"], drop_first=True)
), family=sm.families.Binomial())

result = model.fit()

# Extract human coefficient
# In this design matrix, the human coefficient name is 'is_human'
coef = result.params.get("is_human", np.nan)
se = result.bse.get("is_human", np.nan)

# Wald z test
z = coef / se if se and not np.isnan(se) else np.nan
# Two-sided p-value
p = 2 * (1 - stats.norm.cdf(abs(z))) if not np.isnan(z) else np.nan

# Decide yes/no: positive coef and p<0.05
response = "Yes" if (coef > 0) and (p < 0.05) else "No"

# Map to 0-100 scale
# Use z magnitude with tanh to avoid extremes, center at 50
if np.isnan(z):
    scale = 50
else:
    strength = np.tanh(abs(z) / 3.0)
    if coef > 0:
        scale = int(round(50 + 50 * strength))
    else:
        scale = int(round(50 - 50 * strength))

# Clamp
scale = max(0, min(100, scale))

# Write conclusion
with open("conclusion.txt", "w", encoding="utf-8") as f:
    json.dump({"response": response, "scale": scale}, f)

# Print minimal diagnostics for inspection
print({"coef": coef, "se": se, "z": z, "p": p, "response": response, "scale": scale})
