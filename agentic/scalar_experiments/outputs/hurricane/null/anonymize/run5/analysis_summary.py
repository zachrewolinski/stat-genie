import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api as sm


df = pd.read_csv('hurricane.csv')

# log transforms to handle skew
for col in ['feature8', 'feature9', 'feature14']:
    df[f'log_{col}'] = np.log1p(df[col])

# correlations
corr_fem = df['feature4'].corr(df['log_feature8'])
corr_bin = df['feature6'].corr(df['log_feature8'])

# OLS with controls
ols = smf.ols('log_feature8 ~ feature4 + log_feature14 + feature5 + feature13 + feature7', data=df).fit()
ols_bin = smf.ols('log_feature8 ~ feature6 + log_feature14 + feature5 + feature13 + feature7', data=df).fit()

# Interaction with wind speed (severity)
ols_int = smf.ols('log_feature8 ~ feature4 * feature13 + log_feature14 + feature5 + feature7', data=df).fit()

# Poisson and dispersion
poisson = smf.glm('feature8 ~ feature4 + log_feature14 + feature5 + feature13 + feature7', data=df, family=sm.families.Poisson()).fit()
pearson_chi2 = poisson.pearson_chi2
ratio = pearson_chi2 / poisson.df_resid

# Negative binomial
nb = smf.glm('feature8 ~ feature4 + log_feature14 + feature5 + feature13 + feature7', data=df, family=sm.families.NegativeBinomial()).fit()

results = {
    'corr_fem_logfatal': corr_fem,
    'corr_bin_logfatal': corr_bin,
    'ols_coef_fem': ols.params['feature4'],
    'ols_p_fem': ols.pvalues['feature4'],
    'ols_bin_coef': ols_bin.params['feature6'],
    'ols_bin_p': ols_bin.pvalues['feature6'],
    'ols_int_coef': ols_int.params['feature4:feature13'],
    'ols_int_p': ols_int.pvalues['feature4:feature13'],
    'poisson_coef_fem': poisson.params['feature4'],
    'poisson_p_fem': poisson.pvalues['feature4'],
    'poisson_dispersion_ratio': ratio,
    'nb_coef_fem': nb.params['feature4'],
    'nb_p_fem': nb.pvalues['feature4'],
}

for k, v in results.items():
    print(f"{k}: {v}")
