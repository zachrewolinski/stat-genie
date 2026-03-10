import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api as sm

# Load data
path = "panda_nuts.csv"
df = pd.read_csv(path)

# Compute efficiency as nuts opened per second
# Avoid division issues; seconds should be >0 per metadata

df = df.copy()
df["efficiency"] = df["nuts_opened"] / df["seconds"]

# Keep relevant columns and drop missing
cols = ["efficiency", "age", "sex", "help", "chimpanzee"]
sub = df[cols].dropna()

# OLS with chimpanzee-clustered standard errors
model = smf.ols("efficiency ~ age + C(sex) + C(help)", data=sub).fit(
    cov_type="cluster", cov_kwds={"groups": sub["chimpanzee"]}
)

# Poisson model for counts with offset for time (rate model)
sub_counts = df[["nuts_opened", "seconds", "age", "sex", "help", "chimpanzee"]].dropna()
sub_counts = sub_counts.copy()
sub_counts["log_seconds"] = np.log(sub_counts["seconds"])
poisson = smf.glm(
    "nuts_opened ~ age + C(sex) + C(help)",
    data=sub_counts,
    family=sm.families.Poisson(),
    offset=sub_counts["log_seconds"],
).fit(cov_type="cluster", cov_kwds={"groups": sub_counts["chimpanzee"]})

# Also compute mean efficiency by groups for context
mean_by_sex = sub.groupby("sex")["efficiency"].mean()
mean_by_help = sub.groupby("help")["efficiency"].mean()

# Save key outputs for interpretation
results = {
    "n": int(sub.shape[0]),
    "n_chimps": int(sub["chimpanzee"].nunique()),
    "coef": model.params.to_dict(),
    "pvalues": model.pvalues.to_dict(),
    "conf_int": model.conf_int().to_dict(),
    "r2": float(model.rsquared),
    "poisson_coef": poisson.params.to_dict(),
    "poisson_pvalues": poisson.pvalues.to_dict(),
    "poisson_conf_int": poisson.conf_int().to_dict(),
    "poisson_deviance": float(poisson.deviance),
    "poisson_df_resid": float(poisson.df_resid),
    "mean_efficiency": float(sub["efficiency"].mean()),
    "mean_by_sex": mean_by_sex.to_dict(),
    "mean_by_help": mean_by_help.to_dict(),
}

# Print for the CLI to read
print(results)
