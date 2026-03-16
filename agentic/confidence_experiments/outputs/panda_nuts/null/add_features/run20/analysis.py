import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

DATA_PATH = "panda_nuts.csv"

df = pd.read_csv(DATA_PATH)

# Basic cleaning
if df['help'].dtype == object:
    df['help'] = df['help'].str.strip().str.lower()
if df['sex'].dtype == object:
    df['sex'] = df['sex'].str.strip().str.lower()

key_cols = ['nuts_opened', 'seconds', 'age', 'sex', 'help']
df_clean = df.dropna(subset=key_cols).copy()

df_clean = df_clean[df_clean['seconds'] > 0].copy()

# Rate for descriptive context
df_clean['rate'] = df_clean['nuts_opened'] / df_clean['seconds']

formula = 'nuts_opened ~ age + C(sex) + C(help)'

# Poisson GLM with offset for exposure
poisson_model = smf.glm(
    formula=formula,
    data=df_clean,
    family=sm.families.Poisson(),
    offset=np.log(df_clean['seconds'])
).fit()

# Robust (sandwich) SEs to account for overdispersion
# In this statsmodels version, _get_robustcov_results updates in place and returns None.
poisson_model._get_robustcov_results(cov_type="HC0")
robust_cov = poisson_model.cov_params()
robust_se = np.sqrt(np.diag(robust_cov))
robust_z = poisson_model.params / robust_se
robust_pvalues = 2 * stats.norm.sf(np.abs(robust_z))

# Check dispersion: deviance / df_resid
if poisson_model.df_resid > 0:
    dispersion = float(poisson_model.deviance / poisson_model.df_resid)
else:
    dispersion = np.nan

# Negative Binomial (discrete) with exposure, estimate alpha
nb_model = smf.negativebinomial(
    formula=formula,
    data=df_clean,
    exposure=df_clean['seconds']
).fit(disp=False)

# Likelihood ratio test vs null Poisson model
null_model = smf.glm(
    formula='nuts_opened ~ 1',
    data=df_clean,
    family=sm.families.Poisson(),
    offset=np.log(df_clean['seconds'])
).fit()

lr_stat = 2 * (poisson_model.llf - null_model.llf)
lr_df = poisson_model.df_model
lr_pvalue = stats.chi2.sf(lr_stat, lr_df) if lr_df > 0 else np.nan

print("Rows used:", len(df_clean))
print("Dispersion (Poisson deviance/df):", dispersion)
print("Poisson GLM summary (robust HC0):")
print(poisson_model.summary())
print("\nPoisson GLM robust p-values (HC0):")
for name, pval in zip(poisson_model.params.index, robust_pvalues):
    print(f"{name}: {pval:.6f}")
print("\nLR test vs null (Poisson): stat=%.4f df=%d p=%.6f" % (lr_stat, lr_df, lr_pvalue))
print("\nNegative Binomial (discrete) summary:")
print(nb_model.summary())

# Compute rate ratios from Poisson model (for interpretation)
coef = poisson_model.params
conf = poisson_model.conf_int()
rate_ratios = np.exp(coef)
rr_conf = np.exp(conf)

print("\nRate ratios (Poisson):")
for name in rate_ratios.index:
    print(
        f"{name}: RR={rate_ratios[name]:.3f} "
        f"(95% CI {rr_conf.loc[name, 0]:.3f}, {rr_conf.loc[name, 1]:.3f})"
    )

# Save key results
import json

results = {
    "n": int(len(df_clean)),
    "dispersion": float(dispersion),
    "lr_stat": float(lr_stat),
    "lr_df": int(lr_df),
    "lr_pvalue": float(lr_pvalue),
    "poisson_params": poisson_model.params.to_dict(),
    "poisson_pvalues": poisson_model.pvalues.to_dict(),
    "poisson_robust_pvalues": dict(zip(poisson_model.params.index, robust_pvalues)),
    "poisson_conf_int": conf.rename(columns={0: "lower", 1: "upper"}).to_dict(orient="index"),
    "rate_ratios": rate_ratios.to_dict(),
    "rate_ratio_conf": rr_conf.rename(columns={0: "lower", 1: "upper"}).to_dict(orient="index"),
    "nb_params": nb_model.params.to_dict(),
    "nb_pvalues": nb_model.pvalues.to_dict(),
}

with open("analysis_results.json", "w") as f:
    json.dump(results, f, indent=2)
