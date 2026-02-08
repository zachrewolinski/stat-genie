import pandas as pd
import numpy as np
import statsmodels.api as sm
import patsy
from scipy.stats import norm

# Load data
path = "amtl.csv"
df = pd.read_csv(path)

# Rename for clarity
col_tooth = "feature1"  # tooth class
col_id = "feature2"
col_missing = "feature3"  # missing count
col_obs = "feature4"  # observable sockets
col_age = "feature5"
col_sex = "feature7"
col_genus = "feature8"

# Basic cleaning
for col in [col_missing, col_obs, col_age, col_sex]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=[col_missing, col_obs, col_age, col_sex, col_tooth, col_genus])

# Ensure valid counts
df = df[(df[col_obs] >= 0) & (df[col_missing] >= 0)]
df = df[df[col_missing] <= df[col_obs]]

# Binary indicator for humans
human_label = "Homo sapiens"
df["is_human"] = (df[col_genus] == human_label).astype(int)

# Prepare binomial response
present = df[col_obs] - df[col_missing]
endog = np.column_stack([df[col_missing].values, present.values])

# Design matrix with categorical tooth class
formula = "is_human + C({}) + {} + {}".format(col_tooth, col_age, col_sex)
X = patsy.dmatrix(formula, df, return_type="dataframe")

# Fit GLM binomial
model = sm.GLM(endog, X, family=sm.families.Binomial())
res = model.fit()

# Extract human effect
coef = res.params.get("is_human", np.nan)
se = res.bse.get("is_human", np.nan)

# Compute z and p-value
z = coef / se if np.isfinite(coef) and np.isfinite(se) and se != 0 else np.nan
p = 2 * (1 - norm.cdf(abs(z))) if np.isfinite(z) else np.nan

# Marginal predicted probabilities for human vs non-human
# Use observed covariates, toggle is_human
X_human = X.copy()
X_human["is_human"] = 1
X_non = X.copy()
X_non["is_human"] = 0

pred_human = res.predict(X_human)
pred_non = res.predict(X_non)

avg_human = float(np.mean(pred_human))
avg_non = float(np.mean(pred_non))

diff = avg_human - avg_non

# Print diagnostics
print("N rows:", len(df))
print("Human rows:", int(df["is_human"].sum()))
print("Coef is_human:", coef)
print("SE:", se)
print("z:", z)
print("p:", p)
print("Avg pred human:", avg_human)
print("Avg pred non-human:", avg_non)
print("Diff:", diff)

# Compute scalar conclusion
# Heuristic: sign and magnitude of diff, strength adjusted by p-value
# scale: diff 0.00 -> 0, diff >= 0.10 -> strong; p-value < 0.001 boosts

def clamp(x, lo, hi):
    return max(lo, min(hi, x))

# Base score from diff
base = diff / 0.10  # 0.10 difference = 1.0
base = clamp(base, -1.0, 1.0)

# Significance factor
if np.isnan(p):
    sig = 0.0
elif p < 0.001:
    sig = 1.0
elif p < 0.01:
    sig = 0.8
elif p < 0.05:
    sig = 0.6
elif p < 0.1:
    sig = 0.4
else:
    sig = 0.2

score = base * sig * 100
score = int(round(clamp(score, -100, 100)))

with open("conclusion.txt", "w") as f:
    f.write(str(score))

print("Conclusion score:", score)
