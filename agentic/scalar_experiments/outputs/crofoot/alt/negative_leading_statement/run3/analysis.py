import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

# Load data

df = pd.read_csv("crofoot.csv")

# Derived variables

df["size_diff"] = df["n_focal"] - df["n_other"]
df["size_ratio"] = df["n_focal"] / df["n_other"]
df["loc_adv"] = df["dist_other"] - df["dist_focal"]  # positive => closer to focal home center

# Standardize predictors for comparability
for col in ["size_diff", "loc_adv"]:
    df[f"{col}_z"] = (df[col] - df[col].mean()) / df[col].std(ddof=0)

# Logistic regression
model = smf.logit("win ~ size_diff_z + loc_adv_z", data=df).fit(disp=False)
print(model.summary())

# Alternative with size_ratio

df["size_ratio_z"] = (df["size_ratio"] - df["size_ratio"].mean()) / df["size_ratio"].std(ddof=0)
model2 = smf.logit("win ~ size_ratio_z + loc_adv_z", data=df).fit(disp=False)
print("\nModel with size_ratio:")
print(model2.summary())

# Simple tests: t-test
wins = df[df["win"] == 1]
losses = df[df["win"] == 0]
for col in ["size_diff", "loc_adv"]:
    tstat, pval = stats.ttest_ind(wins[col], losses[col], equal_var=False)
    print(
        f"{col} mean win {wins[col].mean():.3f} vs loss {losses[col].mean():.3f}, "
        f"t={tstat:.3f}, p={pval:.4f}"
    )

# Logistic regression with interaction
model3 = smf.logit("win ~ size_diff_z * loc_adv_z", data=df).fit(disp=False)
print("\nModel with interaction:")
print(model3.summary())
