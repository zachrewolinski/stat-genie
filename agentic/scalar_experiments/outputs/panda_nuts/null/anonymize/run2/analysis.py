import json
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

# Load data
csv_path = 'panda_nuts.csv'
df = pd.read_csv(csv_path)

# Compute efficiency: nuts per minute
# feature5: number of nuts opened, feature6: duration in seconds
# Avoid division by zero just in case

df = df.copy()
df['eff_per_min'] = df['feature5'] / df['feature6'] * 60.0

# Drop rows with missing values in relevant columns
cols = ['feature1', 'feature2', 'feature3', 'feature7', 'eff_per_min']
df_model = df[cols].dropna()

# Basic descriptive stats
n_obs = len(df_model)

# Encode categories for summaries
sex_counts = df_model['feature3'].value_counts(dropna=False)
help_counts = df_model['feature7'].value_counts(dropna=False)

# Group summaries
mean_by_sex = df_model.groupby('feature3')['eff_per_min'].mean()
mean_by_help = df_model.groupby('feature7')['eff_per_min'].mean()

# Correlation with age
age_corr = df_model['feature2'].corr(df_model['eff_per_min'])

# OLS with cluster-robust SEs by individual ID
# feature1 is individual ID
model = smf.ols('eff_per_min ~ feature2 + C(feature3) + C(feature7)', data=df_model)
results = model.fit(cov_type='cluster', cov_kwds={'groups': df_model['feature1']})

# Extract key results
coef_table = results.summary2().tables[1]

# Wald test for joint significance of predictors (excluding intercept)
# Use robust covariance
try:
    wald = results.wald_test('feature2 = 0, C(feature3)[T.m] = 0, C(feature7)[T.y] = 0')
    wald_stat = float(wald.statistic)
    wald_p = float(wald.pvalue)
    wald_df = int(wald.df_denom) if hasattr(wald, 'df_denom') else None
except Exception:
    wald_stat = None
    wald_p = None
    wald_df = None

output = {
    'n_obs': n_obs,
    'sex_counts': sex_counts.to_dict(),
    'help_counts': help_counts.to_dict(),
    'mean_by_sex': mean_by_sex.to_dict(),
    'mean_by_help': mean_by_help.to_dict(),
    'age_corr': age_corr,
    'r_squared': results.rsquared,
    'adj_r_squared': results.rsquared_adj,
    'coef_table': coef_table.to_dict(),
    'wald_stat': wald_stat,
    'wald_p': wald_p,
    'wald_df': wald_df,
}

with open('analysis_results.json', 'w') as f:
    json.dump(output, f, indent=2)

print(json.dumps(output, indent=2))
