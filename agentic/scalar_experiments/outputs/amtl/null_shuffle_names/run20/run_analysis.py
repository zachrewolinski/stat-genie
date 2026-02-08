import pandas as pd
import numpy as np
import statsmodels.api as sm
import patsy

# Load data
raw = pd.read_csv("amtl.csv")

# Rename columns based on metadata mapping
# sockets -> tooth_class (Anterior/Posterior/Premolar)
# prob_male -> specimen_id
# genus -> num_amtl (count of missing teeth)
# age -> num_sockets (count of observable sockets)
# pop -> age_at_death
# num_amtl -> stdev_age
# stdev_age -> prob_male (sex estimate)
# tooth_class -> genus (Homo sapiens, Pan, Papio, Pongo)
# specimen -> region

df = raw.rename(columns={
    "sockets": "tooth_class",
    "prob_male": "specimen_id",
    "genus": "num_amtl",
    "age": "num_sockets",
    "pop": "age_at_death",
    "num_amtl": "stdev_age",
    "stdev_age": "prob_male",
    "tooth_class": "genus",
    "specimen": "region",
})

# Remove rows where AMTL count exceeds observable sockets
valid_mask = df["num_amtl"] <= df["num_sockets"]
invalid_count = int((~valid_mask).sum())
if invalid_count:
    df = df.loc[valid_mask].copy()

# Binary indicator for modern human vs non-human primates
df["is_human"] = (df["genus"] == "Homo sapiens").astype(int)

# Failures for binomial model
df["failures"] = df["num_sockets"] - df["num_amtl"]

# Build design matrices
formula = "num_amtl + failures ~ is_human + age_at_death + prob_male + C(tooth_class)"
y, X = patsy.dmatrices(formula, data=df, return_type="matrix")

# Fit binomial GLM with logit link
model = sm.GLM(y, X, family=sm.families.Binomial()).fit()

col_names = X.design_info.column_names
idx_is_human = col_names.index("is_human")
coef = float(model.params[idx_is_human])
se = float(model.bse[idx_is_human])
pval = float(model.pvalues[idx_is_human])

# Average marginal effect via counterfactual prediction
cf0 = df.copy()
cf1 = df.copy()
cf0["is_human"] = 0
cf1["is_human"] = 1

X0 = patsy.build_design_matrices([X.design_info], cf0)[0]
X1 = patsy.build_design_matrices([X.design_info], cf1)[0]

pred0 = model.predict(X0)
pred1 = model.predict(X1)
ame = float((pred1 - pred0).mean())

# Observed-group mean AMTL rate for context
obs_rate_human = (df.loc[df["is_human"] == 1, "num_amtl"].sum() / df.loc[df["is_human"] == 1, "num_sockets"].sum())
obs_rate_nonhuman = (df.loc[df["is_human"] == 0, "num_amtl"].sum() / df.loc[df["is_human"] == 0, "num_sockets"].sum())

print("n_rows", len(df))
print("invalid_dropped", invalid_count)
print("human_share", df["is_human"].mean())
print("coef_is_human", coef)
print("se_is_human", se)
print("p_is_human", pval)
print("ame_is_human", ame)
print("obs_rate_human", obs_rate_human)
print("obs_rate_nonhuman", obs_rate_nonhuman)
print("model_aic", model.aic)
