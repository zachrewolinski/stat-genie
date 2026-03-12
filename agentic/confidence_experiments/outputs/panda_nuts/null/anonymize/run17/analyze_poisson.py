import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

path = "panda_nuts.csv"
df = pd.read_csv(path)

# Poisson regression on counts with log-duration offset
# Model: nuts opened ~ age + sex + help, offset log(duration)
df = df.copy()

# Avoid zeros in duration; none expected but guard
if (df["feature6"] <= 0).any():
    raise ValueError("Non-positive durations present")

df["log_duration"] = np.log(df["feature6"])

poisson_model = smf.glm(
    "feature5 ~ feature2 + C(feature3) + C(feature7)",
    data=df,
    family=sm.families.Poisson(),
    offset=df["log_duration"],
).fit()

print(poisson_model.summary())

# Check for overdispersion (Pearson chi-square / df)
pearson_chi2 = poisson_model.pearson_chi2
resid_df = poisson_model.df_resid
print("Overdispersion ratio (Pearson chi2 / df):", pearson_chi2 / resid_df)

# Negative binomial if overdispersion is substantial (>1.5)
nb_model = smf.glm(
    "feature5 ~ feature2 + C(feature3) + C(feature7)",
    data=df,
    family=sm.families.NegativeBinomial(),
    offset=df["log_duration"],
).fit()

print(nb_model.summary())
