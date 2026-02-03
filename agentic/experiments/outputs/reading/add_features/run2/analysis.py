import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from statsmodels.stats.weightstats import ttest_ind

# Load data
DF_PATH = 'reading.csv'

df = pd.read_csv(DF_PATH)

# Focus on participants with dyslexia (binary indicator)
sub = df[df['dyslexia_bin'] == 1].copy()

# Basic cleaning
sub = sub[sub['reader_view'].isin([0, 1])]
sub = sub[sub['speed'] > 0]

# Group summary
group_stats = sub.groupby('reader_view')['speed'].agg(['count', 'mean', 'median', 'std'])
print('Group stats (raw speed):')
print(group_stats)

# Welch t-test on raw speed
rv0 = sub[sub['reader_view'] == 0]['speed']
rv1 = sub[sub['reader_view'] == 1]['speed']
print('\nWelch t-test (raw speed):')
print(ttest_ind(rv1, rv0, usevar='unequal'))

# Log-speed analysis for skew
sub['log_speed'] = np.log(sub['speed'])
rv0l = sub[sub['reader_view'] == 0]['log_speed']
rv1l = sub[sub['reader_view'] == 1]['log_speed']
print('\nWelch t-test (log speed):')
print(ttest_ind(rv1l, rv0l, usevar='unequal'))

# Regression controlling for page and word count
# Use log speed to reduce skew
model = smf.ols('log_speed ~ reader_view + num_words + C(page_id)', data=sub).fit(cov_type='HC3')
print('\nOLS (log speed) with controls:')
print(model.summary().tables[1])

coef = model.params.get('reader_view', np.nan)
se = model.bse.get('reader_view', np.nan)
# Convert log-point estimate to percent change
pct_change = (np.exp(coef) - 1) * 100 if pd.notnull(coef) else np.nan
print(f"\nReader view coef (log points): {coef:.6f} (SE {se:.6f}); ~{pct_change:.2f}% change")
