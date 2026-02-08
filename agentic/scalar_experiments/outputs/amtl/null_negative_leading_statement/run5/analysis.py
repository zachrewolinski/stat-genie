import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


df = pd.read_csv("amtl.csv")

# Clean / ensure numeric
for col in ["num_amtl", "sockets", "age", "prob_male"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=["num_amtl", "sockets", "age", "prob_male", "tooth_class", "genus"])

# Binary indicator for modern humans
# Expect genus values include "Homo sapiens"
df["is_human"] = (df["genus"].str.strip() == "Homo sapiens").astype(int)

# Response as two-column for binomial
# Use GLM with binomial family and frequency weights via num_amtl / sockets
# statsmodels formula supports endog as proportion with weights
# Avoid proportion 0/1 issues by using weights and proportion

# Use full model with controls
formula = "amtl_rate ~ is_human + age + prob_male + C(tooth_class)"

df["amtl_rate"] = df["num_amtl"] / df["sockets"]

model = smf.glm(
    formula=formula,
    data=df,
    family=sm.families.Binomial(),
    freq_weights=df["sockets"],
).fit()

coef = model.params.get("is_human", float("nan"))
se = model.bse.get("is_human", float("nan"))
px = model.pvalues.get("is_human", float("nan"))

# Also compute unadjusted mean rates for sanity
mean_human = df.loc[df["is_human"] == 1, "amtl_rate"].mean()
mean_non = df.loc[df["is_human"] == 0, "amtl_rate"].mean()

print("N:", len(df))
print("Human mean rate:", mean_human)
print("Nonhuman mean rate:", mean_non)
print("GLM coef (log-odds) is_human:", coef)
print("SE:", se)
print("p-value:", px)
print(model.summary())
