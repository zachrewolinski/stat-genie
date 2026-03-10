import json
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats


df = pd.read_csv('panda_nuts.csv')

cols = ['age', 'sex', 'help', 'nuts_opened', 'seconds']
missing_cols = [c for c in cols if c not in df.columns]
if missing_cols:
    raise ValueError(f"Missing columns: {missing_cols}")

sub = df[cols].copy()

# normalize categorical values
sub['sex'] = sub['sex'].astype(str).str.strip().str.lower()
sub['help'] = sub['help'].astype(str).str.strip().str.lower()

# keep valid categories
sub = sub[sub['sex'].isin(['m', 'f']) & sub['help'].isin(['y', 'n'])]

# ensure numeric
sub['age'] = pd.to_numeric(sub['age'], errors='coerce')
sub['nuts_opened'] = pd.to_numeric(sub['nuts_opened'], errors='coerce')
sub['seconds'] = pd.to_numeric(sub['seconds'], errors='coerce')

sub = sub.dropna(subset=['age', 'nuts_opened', 'seconds', 'sex', 'help'])
sub = sub[sub['seconds'] > 0]

sub['rate'] = sub['nuts_opened'] / sub['seconds']
sub['log_seconds'] = np.log(sub['seconds'])

# Poisson GLM with offset (rate model)
poisson_model = smf.glm(
    'nuts_opened ~ age + C(sex) + C(help)',
    data=sub,
    family=sm.families.Poisson(),
    offset=sub['log_seconds']
).fit()

# Robust covariance (HC3) for Poisson model
poisson_robust = smf.glm(
    'nuts_opened ~ age + C(sex) + C(help)',
    data=sub,
    family=sm.families.Poisson(),
    offset=sub['log_seconds']
).fit(cov_type='HC3')

# dispersion check
pearson_chi2 = float((poisson_model.resid_pearson ** 2).sum())
dispersion = pearson_chi2 / poisson_model.df_resid

# Negative binomial model (handles overdispersion)
nb_model = smf.glm(
    'nuts_opened ~ age + C(sex) + C(help)',
    data=sub,
    family=sm.families.NegativeBinomial(alpha=1.0),
    offset=sub['log_seconds']
).fit()

# LR test for overall predictors (Poisson)
null_model = smf.glm(
    'nuts_opened ~ 1',
    data=sub,
    family=sm.families.Poisson(),
    offset=sub['log_seconds']
).fit()

lr_stat = 2 * (poisson_model.llf - null_model.llf)
p_lr = stats.chi2.sf(lr_stat, df=poisson_model.df_model)

# Rate ratios (NB)
params = nb_model.params
conf = nb_model.conf_int()
rate_ratios = np.exp(params)
rr_ci_low = np.exp(conf[0])
rr_ci_high = np.exp(conf[1])

# group means
mean_rates = sub.groupby('sex')['rate'].mean().to_dict()
mean_rates_help = sub.groupby('help')['rate'].mean().to_dict()

results = {
    'n': int(len(sub)),
    'dispersion': float(dispersion),
    'poisson_pvalues': poisson_model.pvalues.to_dict(),
    'poisson_robust_pvalues': poisson_robust.pvalues.to_dict(),
    'lr_stat': float(lr_stat),
    'lr_p': float(p_lr),
    'nb_params': params.to_dict(),
    'nb_pvalues': nb_model.pvalues.to_dict(),
    'nb_rate_ratios': rate_ratios.to_dict(),
    'nb_rr_ci_low': rr_ci_low.to_dict(),
    'nb_rr_ci_high': rr_ci_high.to_dict(),
    'mean_rates_by_sex': mean_rates,
    'mean_rates_by_help': mean_rates_help,
}

with open('analysis_results.json', 'w') as f:
    json.dump(results, f, indent=2)
