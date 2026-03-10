import json
import pandas as pd
import numpy as np
import scipy.stats as stats
import statsmodels.formula.api as smf

# Load data
path = 'teachingratings.csv'
df = pd.read_csv(path)

# Identify columns
categorical_cols = ['minority', 'gender', 'credits', 'division', 'native', 'tenure']
for c in categorical_cols:
    if c in df.columns:
        df[c] = df[c].astype('category')

# Basic stats
n = len(df)

# Pearson correlation between beauty and eval
pearson_r, pearson_p = stats.pearsonr(df['beauty'], df['eval'])

# Simple OLS
model_simple = smf.ols('eval ~ beauty', data=df).fit(cov_type='HC3')

# Multivariate OLS with controls
controls = ['age', 'gender', 'minority', 'native', 'tenure', 'division', 'credits', 'students', 'allstudents']
formula = 'eval ~ beauty + ' + ' + '.join(controls)
model_controls = smf.ols(formula, data=df).fit(cov_type='HC3')

# Cluster-robust by professor id if available
if 'prof' in df.columns:
    model_controls_cluster = smf.ols(formula, data=df).fit(cov_type='cluster', cov_kwds={'groups': df['prof']})
else:
    model_controls_cluster = None

# Collect results
sd_beauty = df['beauty'].std()
coef_simple = model_simple.params['beauty']
coef_controls = model_controls.params['beauty']

results = {
    'n': int(n),
    'pearson_r': float(pearson_r),
    'pearson_p': float(pearson_p),
    'simple_coef': float(coef_simple),
    'simple_p': float(model_simple.pvalues['beauty']),
    'controls_coef': float(coef_controls),
    'controls_p': float(model_controls.pvalues['beauty']),
    'controls_r2': float(model_controls.rsquared),
    'sd_beauty': float(sd_beauty),
    'controls_cluster_coef': float(model_controls_cluster.params['beauty']) if model_controls_cluster is not None else None,
    'controls_cluster_p': float(model_controls_cluster.pvalues['beauty']) if model_controls_cluster is not None else None,
}

print(json.dumps(results, indent=2))
