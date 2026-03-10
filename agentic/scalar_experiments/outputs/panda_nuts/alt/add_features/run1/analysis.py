import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data

df = pd.read_csv('panda_nuts.csv')

# Keep relevant columns
# Clean help and sex coding
# help values may be 'y' or 'N'

df['help_bin'] = df['help'].str.strip().str.lower().map({'y': 1, 'n': 0})
df['sex_bin'] = df['sex'].str.strip().str.lower().map({'f': 1, 'm': 0})

# Efficiency as rate

df['efficiency'] = df['nuts_opened'] / df['seconds']

# Drop rows with missing key fields
key_cols = ['nuts_opened', 'seconds', 'age', 'sex_bin', 'help_bin', 'chimpanzee']
df_model = df.dropna(subset=key_cols).copy()

# Ensure positive seconds
# Filter out non-positive if any

df_model = df_model[df_model['seconds'] > 0]

# Offset for rate (log of exposure)

df_model['log_seconds'] = np.log(df_model['seconds'])

results = {}

# GEE Poisson with cluster by chimpanzee
try:
    gee_model = sm.GEE.from_formula(
        'nuts_opened ~ age + sex_bin + help_bin',
        groups='chimpanzee',
        data=df_model,
        family=sm.families.Poisson(),
        offset=df_model['log_seconds']
    )
    gee_res = gee_model.fit()
    results['gee'] = gee_res
except Exception as exc:
    results['gee_error'] = str(exc)

# GLM Poisson with cluster-robust SE by chimpanzee (fallback/robustness)
try:
    glm_model = sm.GLM.from_formula(
        'nuts_opened ~ age + sex_bin + help_bin',
        data=df_model,
        family=sm.families.Poisson(),
        offset=df_model['log_seconds']
    )
    glm_res = glm_model.fit(cov_type='cluster', cov_kwds={'groups': df_model['chimpanzee']})
    results['glm_cluster'] = glm_res
except Exception as exc:
    results['glm_cluster_error'] = str(exc)

# OLS on efficiency (rate) with cluster-robust SE by chimpanzee
try:
    ols_model = sm.OLS.from_formula(
        'efficiency ~ age + sex_bin + help_bin',
        data=df_model
    )
    ols_res = ols_model.fit(cov_type='cluster', cov_kwds={'groups': df_model['chimpanzee']})
    results['ols_cluster'] = ols_res
except Exception as exc:
    results['ols_cluster_error'] = str(exc)

# Summaries
print('N rows used:', len(df_model))
print('N chimpanzees:', df_model['chimpanzee'].nunique())

if 'gee' in results:
    print('\nGEE Poisson (rate with offset)')
    print(results['gee'].summary())
else:
    print('GEE error:', results.get('gee_error'))

if 'glm_cluster' in results:
    print('\nGLM Poisson with cluster-robust SE')
    print(results['glm_cluster'].summary())
else:
    print('GLM error:', results.get('glm_cluster_error'))

if 'ols_cluster' in results:
    print('\nOLS efficiency with cluster-robust SE')
    print(results['ols_cluster'].summary())
else:
    print('OLS error:', results.get('ols_cluster_error'))
