import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

DATA_PATH = "panda_nuts.csv"

df = pd.read_csv(DATA_PATH)

# Basic cleaning
# Ensure expected columns
expected = {"age", "sex", "help", "nuts_opened", "seconds"}
missing = expected - set(df.columns)
if missing:
    raise ValueError(f"Missing columns: {missing}")

# Drop rows with missing key values
df = df.dropna(subset=["age", "sex", "help", "nuts_opened", "seconds"]).copy()

# Efficiency: nuts opened per second
# Use rate for descriptive stats
# Use Poisson GLM with offset for inferential model

df["rate"] = df["nuts_opened"] / df["seconds"]

# Poisson GLM with offset log(seconds)
# Model: nuts_opened ~ age + sex + help + offset(log(seconds))
# Use robust (HC0) covariance
formula = "nuts_opened ~ age + C(sex) + C(help)"

glm_full = smf.glm(
    formula=formula,
    data=df,
    family=sm.families.Poisson(),
    offset=np.log(df["seconds"]),
).fit(cov_type="HC0")

# Null model with only intercept
glm_null = smf.glm(
    formula="nuts_opened ~ 1",
    data=df,
    family=sm.families.Poisson(),
    offset=np.log(df["seconds"]),
).fit(cov_type="HC0")

# Likelihood ratio test for overall predictors
lr_stat = 2 * (glm_full.llf - glm_null.llf)
# df difference = number of predictors in full model (excluding intercept)
# Use model df_model
lr_df = int(glm_full.df_model - glm_null.df_model)
# p-value from chi-square
from scipy.stats import chi2
lr_p = chi2.sf(lr_stat, lr_df) if lr_df > 0 else np.nan

# Overdispersion check (Pearson chi2 / df_resid)
pearson_chi2 = glm_full.pearson_chi2
overdispersion = pearson_chi2 / glm_full.df_resid if glm_full.df_resid > 0 else np.nan

# Extract coefficient table
coef_table = glm_full.summary2().tables[1]
coef_table = coef_table.rename(
    columns={"Coef.": "coef", "Std.Err.": "std_err", "z": "z", "P>|z|": "pvalue"}
)

# Rate ratios for interpretability
coef_table["rate_ratio"] = np.exp(coef_table["coef"])

# Also run OLS on rate for a sanity check
ols = smf.ols("rate ~ age + C(sex) + C(help)", data=df).fit(cov_type="HC0")

# Negative binomial (discrete) to account for overdispersion
nb_full = None
nb_null = None
nb_lr_stat = None
nb_lr_p = None
try:
    nb_full = smf.negativebinomial(
        formula=formula, data=df, offset=np.log(df["seconds"])
    ).fit(disp=False)
    nb_null = smf.negativebinomial(
        formula="nuts_opened ~ 1", data=df, offset=np.log(df["seconds"])
    ).fit(disp=False)
    nb_lr_stat = 2 * (nb_full.llf - nb_null.llf)
    nb_lr_df = int(nb_full.df_model - nb_null.df_model)
    nb_lr_p = chi2.sf(nb_lr_stat, nb_lr_df) if nb_lr_df > 0 else np.nan
except Exception:
    pass

# Build summary
summary = {
    "n_rows": int(len(df)),
    "rate_mean": float(df["rate"].mean()),
    "rate_std": float(df["rate"].std(ddof=1)),
    "glm_full": {
        "aic": float(glm_full.aic),
        "llf": float(glm_full.llf),
        "overdispersion": float(overdispersion),
        "lr_stat": float(lr_stat),
        "lr_df": int(lr_df),
        "lr_p": float(lr_p) if not np.isnan(lr_p) else None,
        "coef_table": coef_table[["coef", "std_err", "z", "pvalue", "rate_ratio"]].to_dict(),
    },
    "ols": {
        "rsquared": float(ols.rsquared),
        "f_pvalue": float(ols.f_pvalue) if ols.f_pvalue is not None else None,
    },
    "neg_binom": None,
}

if nb_full is not None:
    nb_table = nb_full.summary2().tables[1]
    nb_table = nb_table.rename(
        columns={"Coef.": "coef", "Std.Err.": "std_err", "z": "z", "P>|z|": "pvalue"}
    )
    nb_table["rate_ratio"] = np.exp(nb_table["coef"])
    summary["neg_binom"] = {
        "llf": float(nb_full.llf),
        "alpha": float(nb_full.params.get("alpha", np.nan)),
        "lr_stat": float(nb_lr_stat) if nb_lr_stat is not None else None,
        "lr_p": float(nb_lr_p) if nb_lr_p is not None and not np.isnan(nb_lr_p) else None,
        "coef_table": nb_table[["coef", "std_err", "z", "pvalue", "rate_ratio"]].to_dict(),
    }

with open("analysis_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print(json.dumps(summary, indent=2))
