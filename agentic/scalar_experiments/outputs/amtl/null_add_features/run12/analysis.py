import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm
from scipy import stats

# Load data
path = "amtl.csv"
df = pd.read_csv(path)

# Keep relevant columns
needed = ["num_amtl", "sockets", "age", "prob_male", "genus", "tooth_class"]
df = df[needed].copy()

# Drop rows with missing or invalid values
for col in ["num_amtl", "sockets", "age", "prob_male"]:
    df = df[pd.notnull(df[col])]

df = df[df["sockets"] > 0]

df["genus"] = df["genus"].astype(str)

df["tooth_class"] = df["tooth_class"].astype(str)

# Indicator for modern humans
human_label = "Homo sapiens"
df["is_human"] = (df["genus"] == human_label).astype(int)

# Binomial GLM with proportion response and weights = sockets
formula = "amtl_rate ~ is_human + age + prob_male + C(tooth_class)"

df["amtl_rate"] = df["num_amtl"] / df["sockets"]

model = smf.glm(
    formula=formula,
    data=df,
    family=sm.families.Binomial(),
    freq_weights=df["sockets"],
)
result = model.fit()

# Extract human effect
coef = result.params["is_human"]
se = result.bse["is_human"]

# Compute z and p-value
z = coef / se
p = 2 * (1 - stats.norm.cdf(abs(z)))

# Compute average marginal effect of human status
base = df.copy()
base["is_human"] = 0
pred0 = result.predict(base)
base["is_human"] = 1
pred1 = result.predict(base)

marginal_effect = (pred1 - pred0).mean()

summary = {
    "n": int(df.shape[0]),
    "coef_logit": float(coef),
    "se": float(se),
    "z": float(z),
    "p": float(p),
    "avg_marginal_effect": float(marginal_effect),
}

print(summary)
