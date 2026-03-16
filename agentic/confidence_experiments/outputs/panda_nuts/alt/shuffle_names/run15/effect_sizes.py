import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm
import statsmodels.stats.anova as anova


df = pd.read_csv("panda_nuts.csv")

df["age_years"] = df["hammer"].astype(float)
df["sex_mf"] = df["nuts_opened"].astype(str).str.lower().str.strip()
df["help_received"] = df["seconds"].astype(str).str.lower().str.strip()
df["nuts_opened_count"] = df["help"].astype(float)
df["duration_seconds"] = df["chimpanzee"].astype(float)

df = df[df["duration_seconds"] > 0].copy()
df["efficiency"] = df["nuts_opened_count"] / df["duration_seconds"]

model = smf.ols("efficiency ~ age_years + C(sex_mf) + C(help_received)", data=df).fit()
anova_table = anova.anova_lm(model, typ=2)
residual_ss = anova_table.loc["Residual", "sum_sq"]

partial_eta = {}
for term in ["age_years", "C(sex_mf)", "C(help_received)"]:
    ss = anova_table.loc[term, "sum_sq"]
    partial_eta[term] = ss / (ss + residual_ss)

print(anova_table)
print("partial_eta", partial_eta)
