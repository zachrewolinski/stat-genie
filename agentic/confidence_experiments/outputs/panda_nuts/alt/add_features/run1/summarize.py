import pandas as pd
import numpy as np
import statsmodels.api as sm

# Load data

df = pd.read_csv('panda_nuts.csv')

# Clean coding

df['help_bin'] = df['help'].str.strip().str.lower().map({'y': 1, 'n': 0})
df['sex_bin'] = df['sex'].str.strip().str.lower().map({'f': 1, 'm': 0})

# Efficiency rate

df['efficiency'] = df['nuts_opened'] / df['seconds']

# Filter and set offset
key_cols = ['nuts_opened', 'seconds', 'age', 'sex_bin', 'help_bin', 'chimpanzee']
df_model = df.dropna(subset=key_cols).copy()
df_model = df_model[df_model['seconds'] > 0]
df_model['log_seconds'] = np.log(df_model['seconds'])

gee_model = sm.GEE.from_formula(
    'nuts_opened ~ age + sex_bin + help_bin',
    groups='chimpanzee',
    data=df_model,
    family=sm.families.Poisson(),
    offset=df_model['log_seconds']
)
gee_res = gee_model.fit()

# OLS as robustness
ols_model = sm.OLS.from_formula(
    'efficiency ~ age + sex_bin + help_bin',
    data=df_model
)
ols_res = ols_model.fit(cov_type='cluster', cov_kwds={'groups': df_model['chimpanzee']})

print('rows_used', len(df_model))
print('chimpanzees', df_model['chimpanzee'].nunique())

# Rate ratios and CI
params = gee_res.params
conf = gee_res.conf_int()

for name in ['age', 'sex_bin', 'help_bin']:
    coef = params[name]
    rr = float(np.exp(coef))
    ci_low = float(np.exp(conf.loc[name, 0]))
    ci_high = float(np.exp(conf.loc[name, 1]))
    pval = float(gee_res.pvalues[name])
    print(name, 'coef', round(coef, 4), 'rr', round(rr, 3), 'ci', (round(ci_low, 3), round(ci_high, 3)), 'p', round(pval, 4))

print('\nOLS efficiency coefficients')
for name in ['age', 'sex_bin', 'help_bin']:
    coef = float(ols_res.params[name])
    pval = float(ols_res.pvalues[name])
    ci_low, ci_high = ols_res.conf_int().loc[name]
    print(name, 'coef', round(coef, 4), 'ci', (round(ci_low, 4), round(ci_high, 4)), 'p', round(pval, 4))
