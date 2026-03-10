import json
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

# Load data
csv_path = 'teachingratings.csv'
df = pd.read_csv(csv_path)

# Basic cleaning
# Ensure column names are as expected

# Convert categorical variables to category dtype
cat_cols = ['minority', 'gender', 'credits', 'division', 'native', 'tenure']
for col in cat_cols:
    if col in df.columns:
        df[col] = df[col].astype('category')

# Drop rows with missing values in variables we will use
model_cols = ['eval', 'beauty', 'age', 'gender', 'minority', 'credits', 'division', 'native', 'tenure', 'students', 'allstudents', 'prof']
model_cols = [c for c in model_cols if c in df.columns]
model_df = df[model_cols].dropna().copy()

# Simple correlation
corr = model_df['eval'].corr(model_df['beauty'])

# Simple OLS: eval ~ beauty
m1 = smf.ols('eval ~ beauty', data=model_df).fit()

# OLS with controls
# Use C() for categorical variables
formula_parts = ['beauty', 'age', 'C(gender)', 'C(minority)', 'C(credits)', 'C(division)', 'C(native)', 'C(tenure)', 'students', 'allstudents']
formula = 'eval ~ ' + ' + '.join(formula_parts)

m2 = smf.ols(formula, data=model_df).fit()

# Cluster-robust SE by prof (if prof exists)
if 'prof' in model_df.columns:
    m1_cluster = m1.get_robustcov_results(cov_type='cluster', groups=model_df['prof'])
    m2_cluster = m2.get_robustcov_results(cov_type='cluster', groups=model_df['prof'])
else:
    m1_cluster = None
    m2_cluster = None

# Extract key stats
results = {
    'n_rows': int(model_df.shape[0]),
    'corr_eval_beauty': corr,
    'm1_coef_beauty': m1.params['beauty'],
    'm1_pval_beauty': m1.pvalues['beauty'],
    'm1_ci_beauty': m1.conf_int().loc['beauty'].tolist(),
    'm2_coef_beauty': m2.params['beauty'],
    'm2_pval_beauty': m2.pvalues['beauty'],
    'm2_ci_beauty': m2.conf_int().loc['beauty'].tolist(),
}

if m1_cluster is not None:
    results['m1_cluster_coef_beauty'] = m1_cluster.params[m1_cluster.model.exog_names.index('beauty')]
    results['m1_cluster_pval_beauty'] = m1_cluster.pvalues[m1_cluster.model.exog_names.index('beauty')]
    ci = m1_cluster.conf_int()[m1_cluster.model.exog_names.index('beauty')]
    results['m1_cluster_ci_beauty'] = [float(ci[0]), float(ci[1])]

if m2_cluster is not None:
    results['m2_cluster_coef_beauty'] = m2_cluster.params[m2_cluster.model.exog_names.index('beauty')]
    results['m2_cluster_pval_beauty'] = m2_cluster.pvalues[m2_cluster.model.exog_names.index('beauty')]
    ci = m2_cluster.conf_int()[m2_cluster.model.exog_names.index('beauty')]
    results['m2_cluster_ci_beauty'] = [float(ci[0]), float(ci[1])]

# Save results to JSON for inspection
with open('analysis_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(json.dumps(results, indent=2))
