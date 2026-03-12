import json
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf


df = pd.read_csv('panda_nuts.csv')

# efficiency: nuts opened per second
# avoid division by zero

df = df.copy()
df['efficiency'] = df['nuts_opened'] / df['seconds']

# clean help to lowercase y/n
if df['help'].dtype == object:
    df['help'] = df['help'].str.strip().str.lower().replace({'n': 'n', 'no': 'n', 'y': 'y', 'yes': 'y'})

# clean sex to lowercase
if df['sex'].dtype == object:
    df['sex'] = df['sex'].str.strip().str.lower()

# drop rows with missing
model_df = df[['efficiency','age','sex','help','chimpanzee']].dropna()

model = smf.ols('efficiency ~ age + C(sex) + C(help)', data=model_df).fit()
robust = model.get_robustcov_results(cov_type='cluster', groups=model_df['chimpanzee'])

param_names = model.params.index.tolist()
params = dict(zip(param_names, robust.params))
pvalues = dict(zip(param_names, robust.pvalues))
conf_int = dict(zip(param_names, robust.conf_int().tolist()))

# joint test of age and categorical terms
hypotheses = [f"{name} = 0" for name in param_names if name != 'Intercept']

if hypotheses:
    f_test = robust.f_test(hypotheses)
    f_stat = float(f_test.fvalue)
    f_pvalue = float(f_test.pvalue)
else:
    f_stat = np.nan
    f_pvalue = np.nan

results = {
    'n': int(model_df.shape[0]),
    'n_chimps': int(model_df['chimpanzee'].nunique()),
    'coefficients': {name: float(val) for name, val in params.items()},
    'pvalues': {name: float(val) for name, val in pvalues.items()},
    'conf_int': {name: [float(x) for x in ci] for name, ci in conf_int.items()},
    'r2': float(model.rsquared),
    'adj_r2': float(model.rsquared_adj),
    'f_stat_all': f_stat,
    'f_pvalue_all': f_pvalue,
    'efficiency_mean': float(model_df['efficiency'].mean()),
    'efficiency_std': float(model_df['efficiency'].std()),
}

print(json.dumps(results, indent=2))
