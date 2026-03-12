import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data

df = pd.read_csv('panda_nuts.csv')

# Basic derived metrics

df['rate'] = df['nuts_opened'] / df['seconds']

# Ensure categorical

df['sex'] = df['sex'].astype('category')

df['help'] = df['help'].astype('category')

# Poisson GLM with offset for exposure (seconds)

formula = 'nuts_opened ~ age + C(sex) + C(help)'

model_pois = smf.glm(formula=formula, data=df, family=sm.families.Poisson(), offset=np.log(df['seconds']))
res_pois = model_pois.fit(cov_type='cluster', cov_kwds={'groups': df['chimpanzee']})

# Negative Binomial GLM (to check overdispersion)

model_nb = smf.glm(formula=formula, data=df, family=sm.families.NegativeBinomial(alpha=1.0), offset=np.log(df['seconds']))
res_nb = model_nb.fit(cov_type='cluster', cov_kwds={'groups': df['chimpanzee']})

# Linear model on rate (OLS) with cluster-robust SEs

model_ols = smf.ols('rate ~ age + C(sex) + C(help)', data=df)
res_ols = model_ols.fit(cov_type='cluster', cov_kwds={'groups': df['chimpanzee']})

# Compute descriptive stats

summary = {
    'n_rows': len(df),
    'n_chimps': df['chimpanzee'].nunique(),
    'rate_mean': df['rate'].mean(),
    'rate_sd': df['rate'].std(),
    'age_corr_rate': df['age'].corr(df['rate']),
    'rate_by_sex': df.groupby('sex')['rate'].mean().to_dict(),
    'rate_by_help': df.groupby('help')['rate'].mean().to_dict(),
}

print('DESCRIPTIVES', summary)
print('\nPOISSON (cluster-robust)')
print(res_pois.summary())
print('\nNEG BIN (cluster-robust)')
print(res_nb.summary())
print('\nOLS RATE (cluster-robust)')
print(res_ols.summary())

# Export key stats to CSV for easy reference

out = pd.DataFrame({
    'param': res_pois.params.index,
    'coef_pois': res_pois.params.values,
    'pval_pois': res_pois.pvalues.values,
    'coef_nb': res_nb.params.values,
    'pval_nb': res_nb.pvalues.values,
    'coef_ols': res_ols.params.values,
    'pval_ols': res_ols.pvalues.values,
})

out.to_csv('model_summary.csv', index=False)

print('\nSaved model_summary.csv')
