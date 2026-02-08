import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data

df = pd.read_csv("amtl.csv")

# Basic cleaning
# Keep rows with valid sockets and num_amtl

df = df.dropna(subset=["num_amtl", "sockets", "age", "prob_male", "tooth_class", "genus"])
# Filter out any zero or negative sockets

df = df[df["sockets"] > 0].copy()

# Binary indicator for modern humans

df["human"] = (df["genus"] == "Homo sapiens").astype(int)

# Proportion outcome with binomial weights

df["prop_amtl"] = df["num_amtl"] / df["sockets"]

# Fit binomial GLM with logit link

model = smf.glm(
    "prop_amtl ~ human + age + prob_male + C(tooth_class)",
    data=df,
    family=sm.families.Binomial(),
    freq_weights=df["sockets"],
)
result = model.fit()

# Cluster-robust SE by specimen (if possible)

cluster_result = None
if "specimen" in df.columns:
    try:
        cluster_result = model.fit(cov_type="cluster", cov_kwds={"groups": df["specimen"]})
    except Exception:
        cluster_result = None

# Summaries

coef = result.params.get("human", float("nan"))
se = result.bse.get("human", float("nan"))
pval = result.pvalues.get("human", float("nan"))

print("N rows:", len(df))
print("Human rows:", df["human"].sum())
print("Non-human rows:", (1 - df["human"]).sum())
print("GLM (non-robust) human coef:", coef)
print("SE:", se)
print("p-value:", pval)
print("Odds ratio:", float(np.exp(coef)))

if cluster_result is not None:
    ccoef = cluster_result.params.get("human", float("nan"))
    cpval = cluster_result.pvalues.get("human", float("nan"))
    cse = cluster_result.bse.get("human", float("nan"))
    print("GLM (cluster-robust) human coef:", ccoef)
    print("Cluster SE:", cse)
    print("Cluster p-value:", cpval)

# Effect size as predicted difference at mean covariates

mean_age = df["age"].mean()
mean_male = df["prob_male"].mean()
# Most common tooth_class for reference
mode_tooth = df["tooth_class"].mode().iloc[0]

ref = pd.DataFrame(
    {
        "human": [0, 1],
        "age": [mean_age, mean_age],
        "prob_male": [mean_male, mean_male],
        "tooth_class": [mode_tooth, mode_tooth],
    }
)

pred = result.predict(ref)
print("Predicted AMTL proportion at means (non-human, human):", list(pred))
print("Difference (human - non-human):", float(pred.iloc[1] - pred.iloc[0]))
