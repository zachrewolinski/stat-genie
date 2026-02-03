import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

# Load data
_df = pd.read_csv('hurricane.csv')

# Prepare variables
_df['log_deaths'] = np.log1p(_df['alldeaths'])

# Severity index using standardized wind (higher is worse), minimum pressure (lower is worse), and category
_df['min_inv'] = -_df['min']
for col in ['wind', 'min_inv', 'category']:
    _df[f'z_{col}'] = (_df[col] - _df[col].mean()) / _df[col].std(ddof=0)

_df['severity'] = _df[['z_wind', 'z_min_inv', 'z_category']].mean(axis=1)

# Regression with interaction
model = smf.ols('log_deaths ~ masfem + severity + masfem:severity', data=_df).fit(cov_type='HC3')

# Median splits for descriptive comparisons
median_sev = _df['severity'].median()
median_fem = _df['masfem'].median()
_df['sev_group'] = np.where(_df['severity'] >= median_sev, 'High', 'Low')
_df['fem_group'] = np.where(_df['masfem'] >= median_fem, 'More_feminine', 'Less_feminine')

# Correlations within groups
corr_high = _df.loc[_df['sev_group'] == 'High', ['masfem', 'log_deaths']].corr().iloc[0, 1]
corr_low = _df.loc[_df['sev_group'] == 'Low', ['masfem', 'log_deaths']].corr().iloc[0, 1]

# Group means within high severity storms
high_group = _df[_df['sev_group'] == 'High'].groupby('fem_group')['log_deaths'].mean()
high_mean_more_fem = float(high_group.get('More_feminine', np.nan))
high_mean_less_fem = float(high_group.get('Less_feminine', np.nan))

# Summaries
summary = {
    'n': int(len(_df)),
    'interaction_coef': float(model.params['masfem:severity']),
    'interaction_p': float(model.pvalues['masfem:severity']),
    'masfem_coef': float(model.params['masfem']),
    'masfem_p': float(model.pvalues['masfem']),
    'severity_coef': float(model.params['severity']),
    'severity_p': float(model.pvalues['severity']),
    'corr_high_severity': float(corr_high),
    'corr_low_severity': float(corr_low),
    'high_severity_mean_log_deaths_more_feminine': high_mean_more_fem,
    'high_severity_mean_log_deaths_less_feminine': high_mean_less_fem,
}

print('MODEL_SUMMARY_START')
print(model.summary())
print('MODEL_SUMMARY_END')
print('KEY_RESULTS')
for k, v in summary.items():
    print(f'{k}: {v}')
