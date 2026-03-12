import json
import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.formula.api as smf

# Load data
_df = pd.read_csv('reading.csv')

# Define dyslexia subset
# Use dyslexia_bin==1 for dyslexic participants (includes severe)
df = _df[_df['dyslexia_bin'] == 1].copy()

# Basic counts
n_total = len(df)

# Split by reader_view
rv0 = df[df['reader_view'] == 0]
rv1 = df[df['reader_view'] == 1]

# Descriptive stats
mean0 = rv0['speed'].mean()
mean1 = rv1['speed'].mean()
med0 = rv0['speed'].median()
med1 = rv1['speed'].median()

# Welch t-test
t_stat, p_val = stats.ttest_ind(rv1['speed'], rv0['speed'], equal_var=False, nan_policy='omit')

# Cohen's d (using pooled SD for comparability)
# Use sample SDs
sd0 = rv0['speed'].std(ddof=1)
sd1 = rv1['speed'].std(ddof=1)
# pooled SD
n0 = rv0['speed'].notna().sum()
n1 = rv1['speed'].notna().sum()
pooled_sd = np.sqrt(((n0 - 1) * sd0**2 + (n1 - 1) * sd1**2) / (n0 + n1 - 2)) if (n0 + n1 - 2) > 0 else np.nan
cohen_d = (mean1 - mean0) / pooled_sd if pooled_sd and not np.isnan(pooled_sd) else np.nan

# Regression with log speed to handle skew
# Add small constant to avoid log(0)
_df2 = df.copy()
_df2 = _df2[_df2['speed'] > 0].copy()
_df2['log_speed'] = np.log(_df2['speed'])

# Build formula with key controls and categorical vars
# Keep model parsimonious to avoid overfitting while adjusting for page + device + demographics
formula = (
    "log_speed ~ reader_view + num_words + Flesch_Kincaid + correct_rate + retake_trial + "
    "age + C(gender) + C(device) + C(education) + C(language) + C(page_id) + C(english_native)"
)

# Ensure complete-case data for all variables used in the formula
model_vars = [
    'log_speed', 'reader_view', 'num_words', 'Flesch_Kincaid', 'correct_rate',
    'retake_trial', 'age', 'gender', 'device', 'education', 'language', 'page_id', 'english_native', 'uuid'
]
_df2 = _df2.dropna(subset=model_vars).copy()

model = smf.ols(formula, data=_df2)
res = model.fit(cov_type='cluster', cov_kwds={'groups': _df2['uuid']})

# Extract effect
coef = res.params.get('reader_view', np.nan)
se = res.bse.get('reader_view', np.nan)
p_reg = res.pvalues.get('reader_view', np.nan)

# Convert log-scale coefficient to percent change
pct_change = (np.exp(coef) - 1) * 100 if pd.notna(coef) else np.nan

# Summaries
out = {
    'n_total_dyslexia': int(n_total),
    'n_reader_view_0': int(n0),
    'n_reader_view_1': int(n1),
    'mean_speed_rv0': float(mean0),
    'mean_speed_rv1': float(mean1),
    'median_speed_rv0': float(med0),
    'median_speed_rv1': float(med1),
    'welch_t_p': float(p_val),
    'welch_t_stat': float(t_stat),
    'cohen_d': float(cohen_d),
    'log_speed_coef_reader_view': float(coef),
    'log_speed_se_reader_view': float(se),
    'log_speed_p_reader_view': float(p_reg),
    'log_speed_pct_change_reader_view': float(pct_change),
}

with open('analysis_results.json', 'w') as f:
    json.dump(out, f, indent=2)

print(json.dumps(out, indent=2))
