import json
import pandas as pd
import numpy as np
import scipy.stats as stats
import statsmodels.formula.api as smf

# Load data
path = 'teachingratings.csv'
df = pd.read_csv(path)

# Basic cleaning: drop rows with missing beauty or eval
analysis_df = df.dropna(subset=['beauty', 'eval']).copy()

# Coerce key categoricals
cat_cols = ['minority', 'gender', 'credits', 'division', 'native', 'tenure']
for col in cat_cols:
    if col in analysis_df.columns:
        analysis_df[col] = analysis_df[col].astype('category')

# Summary stats
n_total = len(df)
n_analysis = len(analysis_df)

# Correlations
pearson_r, pearson_p = stats.pearsonr(analysis_df['beauty'], analysis_df['eval'])
spearman_r, spearman_p = stats.spearmanr(analysis_df['beauty'], analysis_df['eval'])

# Simple OLS
model_simple = smf.ols('eval ~ beauty', data=analysis_df).fit()

# Multivariate OLS with controls
# Use typical course/instructor covariates in this dataset
control_vars = ['age', 'gender', 'minority', 'native', 'tenure', 'division', 'credits', 'students']
control_vars = [v for v in control_vars if v in analysis_df.columns]
formula = 'eval ~ beauty'
if control_vars:
    formula += ' + ' + ' + '.join(control_vars)

model_ctrl = smf.ols(formula, data=analysis_df).fit()

# Cluster-robust SE by professor if available
if 'prof' in analysis_df.columns:
    try:
        model_ctrl_cluster = smf.ols(formula, data=analysis_df).fit(cov_type='cluster', cov_kwds={'groups': analysis_df['prof']})
    except Exception:
        model_ctrl_cluster = None
else:
    model_ctrl_cluster = None

# Standardized effect for beauty (simple standardization)
beauty_z = (analysis_df['beauty'] - analysis_df['beauty'].mean()) / analysis_df['beauty'].std(ddof=0)
eval_z = (analysis_df['eval'] - analysis_df['eval'].mean()) / analysis_df['eval'].std(ddof=0)
std_df = analysis_df.copy()
std_df['beauty_z'] = beauty_z
std_df['eval_z'] = eval_z
model_std = smf.ols('eval_z ~ beauty_z', data=std_df).fit()

results = {
    'n_total': n_total,
    'n_analysis': n_analysis,
    'pearson_r': pearson_r,
    'pearson_p': pearson_p,
    'spearman_r': spearman_r,
    'spearman_p': spearman_p,
    'simple_coef': model_simple.params.get('beauty', np.nan),
    'simple_p': model_simple.pvalues.get('beauty', np.nan),
    'simple_ci': model_simple.conf_int().loc['beauty'].tolist() if 'beauty' in model_simple.params else [np.nan, np.nan],
    'ctrl_formula': formula,
    'ctrl_coef': model_ctrl.params.get('beauty', np.nan),
    'ctrl_p': model_ctrl.pvalues.get('beauty', np.nan),
    'ctrl_ci': model_ctrl.conf_int().loc['beauty'].tolist() if 'beauty' in model_ctrl.params else [np.nan, np.nan],
    'cluster_coef': None,
    'cluster_p': None,
    'cluster_ci': None,
    'std_beta': model_std.params.get('beauty_z', np.nan),
    'std_p': model_std.pvalues.get('beauty_z', np.nan),
}

if model_ctrl_cluster is not None and 'beauty' in model_ctrl_cluster.params:
    results['cluster_coef'] = float(model_ctrl_cluster.params['beauty'])
    results['cluster_p'] = float(model_ctrl_cluster.pvalues['beauty'])
    ci = model_ctrl_cluster.conf_int().loc['beauty'].tolist()
    results['cluster_ci'] = [float(ci[0]), float(ci[1])]

with open('analysis_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(json.dumps(results, indent=2))
