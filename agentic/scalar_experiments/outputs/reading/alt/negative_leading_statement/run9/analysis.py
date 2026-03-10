import json
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.formula.api as smf

# Load data
path = 'reading.csv'
df = pd.read_csv(path)

# Focus on individuals with dyslexia (binary indicator)
# dyslexia_bin: 1 indicates dyslexia
sub = df[df['dyslexia_bin'] == 1].copy()

# Basic counts
n_total = len(sub)
counts = sub['reader_view'].value_counts(dropna=False).to_dict()

# Ensure speed positive for log transform
sub = sub[sub['speed'] > 0].copy()
sub['log_speed'] = np.log(sub['speed'])

# Group stats
grp = sub.groupby('reader_view')['speed']
mean_speed = grp.mean()
median_speed = grp.median()

# Welch t-test on log speed (reader_view 1 vs 0)
rv1 = sub[sub['reader_view'] == 1]['log_speed']
rv0 = sub[sub['reader_view'] == 0]['log_speed']

ttest_res = stats.ttest_ind(rv1, rv0, equal_var=False, nan_policy='omit')

# Mann-Whitney U on speed (nonparametric)
try:
    mwu_res = stats.mannwhitneyu(
        sub[sub['reader_view'] == 1]['speed'],
        sub[sub['reader_view'] == 0]['speed'],
        alternative='two-sided'
    )
except Exception as e:
    mwu_res = None

# Effect size (Cohen's d on log speed)
mean1 = rv1.mean()
mean0 = rv0.mean()
var1 = rv1.var(ddof=1)
var0 = rv0.var(ddof=1)
# pooled sd for unequal n
n1 = rv1.shape[0]
n0 = rv0.shape[0]
pooled_sd = np.sqrt(((n1-1)*var1 + (n0-1)*var0) / (n1+n0-2)) if n1+n0-2 > 0 else np.nan
cohen_d = (mean1 - mean0) / pooled_sd if pooled_sd and pooled_sd > 0 else np.nan

# Regression with controls, clustered by uuid
# Use a modest set of controls to avoid overfitting
# Include page_id, num_words, device, age, gender, language, english_native
# Drop rows with missing in these columns
model_df = sub.dropna(subset=['log_speed','reader_view','page_id','num_words','device','age','gender','language','english_native','uuid']).copy()

formula = 'log_speed ~ reader_view + C(page_id) + num_words + C(device) + age + C(gender) + C(language) + C(english_native)'

model = smf.ols(formula=formula, data=model_df).fit(cov_type='cluster', cov_kwds={'groups': model_df['uuid']})

coef = model.params.get('reader_view', np.nan)
pval = model.pvalues.get('reader_view', np.nan)

# Convert log-effect to percent change
pct_change = (np.exp(coef) - 1) * 100 if pd.notnull(coef) else np.nan

result = {
    'n_total_dyslexia_bin1': int(n_total),
    'counts_reader_view': {str(k): int(v) for k, v in counts.items()},
    'mean_speed': {str(k): float(v) for k, v in mean_speed.items()},
    'median_speed': {str(k): float(v) for k, v in median_speed.items()},
    'ttest_log_speed': {
        'stat': float(ttest_res.statistic),
        'pvalue': float(ttest_res.pvalue),
        'n1': int(n1),
        'n0': int(n0),
    },
    'mannwhitney_speed': None if mwu_res is None else {
        'stat': float(mwu_res.statistic),
        'pvalue': float(mwu_res.pvalue)
    },
    'cohen_d_log_speed': float(cohen_d),
    'regression': {
        'coef_reader_view': float(coef),
        'pvalue_reader_view': float(pval),
        'pct_change': float(pct_change),
        'n_obs': int(model_df.shape[0])
    }
}

print(json.dumps(result, indent=2))
