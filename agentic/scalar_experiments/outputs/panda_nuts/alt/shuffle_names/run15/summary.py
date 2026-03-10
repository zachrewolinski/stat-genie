import pandas as pd
import statsmodels.formula.api as smf


df = pd.read_csv("panda_nuts.csv")

df["age_years"] = df["hammer"].astype(float)
df["sex_mf"] = df["nuts_opened"].astype(str).str.lower().str.strip()
df["help_received"] = df["seconds"].astype(str).str.lower().str.strip()
df["nuts_opened_count"] = df["help"].astype(float)
df["duration_seconds"] = df["chimpanzee"].astype(float)

# efficiency

df = df[df["duration_seconds"] > 0].copy()
df["efficiency"] = df["nuts_opened_count"] / df["duration_seconds"]

model = smf.ols("efficiency ~ age_years + C(sex_mf) + C(help_received)", data=df).fit(cov_type="HC3")

print("N", len(df))
print("R2", model.rsquared)

params = model.params
pvals = model.pvalues
print("params", params.to_dict())
print("pvals", pvals.to_dict())

print("means sex", df.groupby("sex_mf")["efficiency"].mean().to_dict())
print("means help", df.groupby("help_received")["efficiency"].mean().to_dict())
print("help counts", df["help_received"].value_counts().to_dict())
print("age corr", df["age_years"].corr(df["efficiency"]))
