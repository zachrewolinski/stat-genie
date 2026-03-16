import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
csv_path = 'panda_nuts.csv'
df = pd.read_csv(csv_path)

# Ensure correct dtypes
for col in ['feature3', 'feature4', 'feature7']:
    df[col] = df[col].astype('category')

# Efficiency rate: nuts per second (not used directly in Poisson model)
df['efficiency'] = df['feature5'] / df['feature6']

# Poisson GLM with log(duration) offset to model nuts opened per unit time
# Predictors: age (feature2), sex (feature3), help (feature7)
model = smf.glm(
    formula='feature5 ~ feature2 + C(feature3) + C(feature7)',
    data=df,
    family=sm.families.Poisson(),
    offset=np.log(df['feature6'])
)
res = model.fit(cov_type='cluster', cov_kwds={'groups': df['feature1']})

# Overdispersion check
pearson_chi2 = sum(res.resid_pearson**2)
dispersion = pearson_chi2 / res.df_resid

# Linear model on efficiency as robustness (with cluster-robust SE)
ols_model = smf.ols('efficiency ~ feature2 + C(feature3) + C(feature7)', data=df)
ols_res = ols_model.fit(cov_type='cluster', cov_kwds={'groups': df['feature1']})

# Collect results
results = {
    'n_rows': int(df.shape[0]),
    'n_ids': int(df['feature1'].nunique()),
    'poisson_params': res.params.to_dict(),
    'poisson_pvalues': res.pvalues.to_dict(),
    'poisson_conf_int': res.conf_int().rename(columns={0:'low',1:'high'}).to_dict(orient='index'),
    'poisson_dispersion': float(dispersion),
    'ols_params': ols_res.params.to_dict(),
    'ols_pvalues': ols_res.pvalues.to_dict(),
    'ols_conf_int': ols_res.conf_int().rename(columns={0:'low',1:'high'}).to_dict(orient='index'),
}

# Compute rate ratios (exp coefficients) for Poisson
rate_ratios = {k: float(np.exp(v)) for k, v in res.params.items()}
results['poisson_rate_ratios'] = rate_ratios

# Save a compact report for inspection
import json
with open('analysis_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print('Wrote analysis_results.json')
