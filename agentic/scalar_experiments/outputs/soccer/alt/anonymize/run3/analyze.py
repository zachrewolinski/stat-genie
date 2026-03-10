import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.weightstats import ztest

# Load data
path = 'soccer.csv'
df = pd.read_csv(path)

# Use rows with skin tone ratings
skin_cols = ['feature18', 'feature19']
work = df.dropna(subset=skin_cols).copy()
work['skin_avg'] = work[skin_cols].mean(axis=1)

# Outcome and exposure
work['red_cards'] = work['feature16']
work['games'] = work['feature9']

# Basic rates by skin tone category
# Define categories: light <=0.25, medium=0.5, dark >=0.75
bins = [-0.01, 0.25, 0.5, 0.75, 1.01]
labels = ['light', 'mid_light', 'mid_dark', 'dark']
work['skin_cat'] = pd.cut(work['skin_avg'], bins=bins, labels=labels)

rate_table = (
    work.groupby('skin_cat')
    .agg(red_cards=('red_cards', 'sum'), games=('games', 'sum'), dyads=('red_cards', 'size'))
    .assign(rate_per_game=lambda x: x['red_cards'] / x['games'])
)

# Poisson regression with offset
X = sm.add_constant(work['skin_avg'])
offset = np.log(work['games'])
poisson_model = sm.GLM(work['red_cards'], X, family=sm.families.Poisson(), offset=offset)
poisson_res = poisson_model.fit()

# Overdispersion check
pearson_chi2 = ((poisson_res.resid_pearson)**2).sum()
pearson_dispersion = pearson_chi2 / poisson_res.df_resid

# Negative binomial regression as robustness check
nb_model = sm.GLM(work['red_cards'], X, family=sm.families.NegativeBinomial(alpha=1.0), offset=offset)
nb_res = nb_model.fit()

# Logistic regression for any red card (including log games as covariate)
work['any_red'] = (work['red_cards'] > 0).astype(int)
X_logit = sm.add_constant(pd.DataFrame({
    'skin_avg': work['skin_avg'],
    'log_games': np.log(work['games'])
}))
logit_model = sm.Logit(work['any_red'], X_logit)
logit_res = logit_model.fit(disp=0)

# Compute effect sizes
# Poisson IRR for skin_avg (per 1.0 unit from 0 to 1 scale)
poisson_coef = poisson_res.params['skin_avg']
poisson_se = poisson_res.bse['skin_avg']
poisson_irr = np.exp(poisson_coef)
poisson_ci = np.exp(poisson_coef + np.array([-1, 1]) * 1.96 * poisson_se)

nb_coef = nb_res.params['skin_avg']
nb_se = nb_res.bse['skin_avg']
nb_irr = np.exp(nb_coef)
nb_ci = np.exp(nb_coef + np.array([-1, 1]) * 1.96 * nb_se)

logit_coef = logit_res.params['skin_avg']
logit_se = logit_res.bse['skin_avg']
logit_or = np.exp(logit_coef)
logit_ci = np.exp(logit_coef + np.array([-1, 1]) * 1.96 * logit_se)

# Compare dark vs light rate ratio (aggregated)
light = rate_table.loc['light']
dark = rate_table.loc['dark']
# Rate ratio for Poisson counts with exposure
rate_ratio = (dark['red_cards']/dark['games']) / (light['red_cards']/light['games'])
# Approximate CI for rate ratio using log method
# var(log(rate)) = 1/red_cards for Poisson; if red_cards=0 handle
rr_ci = (np.nan, np.nan)
if light['red_cards'] > 0 and dark['red_cards'] > 0:
    se_log_rr = np.sqrt(1/dark['red_cards'] + 1/light['red_cards'])
    rr_ci = (
        np.exp(np.log(rate_ratio) - 1.96*se_log_rr),
        np.exp(np.log(rate_ratio) + 1.96*se_log_rr)
    )

# Save summary to stdout
print('Rows with skin ratings:', len(work))
print('\nRate table by skin category:')
print(rate_table)
print('\nPoisson IRR (skin_avg):', poisson_irr)
print('Poisson 95% CI:', poisson_ci)
print('Poisson p-value:', poisson_res.pvalues['skin_avg'])
print('Poisson dispersion (Pearson):', pearson_dispersion)
print('\nNegBin IRR (skin_avg):', nb_irr)
print('NegBin 95% CI:', nb_ci)
print('NegBin p-value:', nb_res.pvalues['skin_avg'])
print('\nLogit OR (skin_avg):', logit_or)
print('Logit 95% CI:', logit_ci)
print('Logit p-value:', logit_res.pvalues['skin_avg'])
print('\nDark vs light rate ratio:', rate_ratio)
print('Dark vs light 95% CI:', rr_ci)
