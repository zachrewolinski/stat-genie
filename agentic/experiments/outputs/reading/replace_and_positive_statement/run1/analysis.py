import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
df = pd.read_csv('reading.csv')

# The dyslexia columns appear noise-injected; derive a binary indicator by rounding
df['dyslexia_bin_round'] = df['dyslexia_bin'].round().clip(lower=0, upper=1).astype(int)
df['dyslexia_round'] = df['dyslexia'].round().clip(lower=0, upper=2).astype(int)

# Focus on participants with dyslexia using the binary indicator
dys = df[df['dyslexia_bin_round'] == 1].copy()

# Basic group stats
group_stats = dys.groupby('reader_view')['speed'].agg(['count', 'mean', 'median', 'std'])

# Log-transform speed for regression due to skew
dys = dys[np.isfinite(dys['speed']) & (dys['speed'] > 0)].copy()
dys['log_speed'] = np.log(dys['speed'])

# Simple regression: log_speed ~ reader_view
model_simple = smf.ols('log_speed ~ reader_view', data=dys).fit()

# Adjusted regression with controls
# Use categorical controls for page_id, device, language, education, english_native
# Include numeric controls: age, num_words, Flesch_Kincaid, correct_rate
formula = (
    'log_speed ~ reader_view + age + num_words + Flesch_Kincaid + correct_rate '
    '+ C(page_id) + C(device) + C(language) + C(education) + C(english_native)'
)
model_adj = smf.ols(formula, data=dys).fit()

# Extract effect size as percent change
def pct_change_from_log(coef):
    return (np.exp(coef) - 1) * 100

simple_coef = model_simple.params['reader_view']
adj_coef = model_adj.params['reader_view']

results = {
    'group_stats': group_stats,
    'simple_coef': simple_coef,
    'simple_pval': model_simple.pvalues['reader_view'],
    'simple_pct_change': pct_change_from_log(simple_coef),
    'adj_coef': adj_coef,
    'adj_pval': model_adj.pvalues['reader_view'],
    'adj_pct_change': pct_change_from_log(adj_coef),
}

# Sensitivity: treat dyslexia as rounded dyslexia >=1 (includes severe)
sens = df[df['dyslexia_round'] >= 1].copy()
sens = sens[np.isfinite(sens['speed']) & (sens['speed'] > 0)].copy()
sens['log_speed'] = np.log(sens['speed'])
sens_group = sens.groupby('reader_view')['speed'].agg(['count', 'mean', 'median', 'std'])
sens_model = smf.ols('log_speed ~ reader_view', data=sens).fit()
sens_coef = sens_model.params['reader_view']
sens_pval = sens_model.pvalues['reader_view']
sens_pct = pct_change_from_log(sens_coef)

print('Group stats (dyslexia only):')
print(group_stats)
print('\nSimple log-speed regression:')
print(f"coef={simple_coef:.4f}, p={results['simple_pval']:.4g}, pct_change={results['simple_pct_change']:.2f}%")
print('\nAdjusted log-speed regression:')
print(f"coef={adj_coef:.4f}, p={results['adj_pval']:.4g}, pct_change={results['adj_pct_change']:.2f}%")

print('\nSensitivity (dyslexia rounded >= 1) group stats:')
print(sens_group)
print('\nSensitivity simple log-speed regression:')
print(f"coef={sens_coef:.4f}, p={sens_pval:.4g}, pct_change={sens_pct:.2f}%")
