import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.weightstats import ttest_ind

# Load data
path = 'reading.csv'
df = pd.read_csv(path)

# Focus on participants with dyslexia
# dyslexia_bin: 1 indicates dyslexia
sub = df[df['dyslexia_bin'] == 1].copy()

# Basic group stats by reader_view
group_stats = sub.groupby('reader_view')['speed'].agg(['count', 'mean', 'median', 'std']).rename_axis('reader_view')

# Welch's t-test for difference in means
speed_rv1 = sub[sub['reader_view'] == 1]['speed'].dropna()
speed_rv0 = sub[sub['reader_view'] == 0]['speed'].dropna()

# If either group is empty, skip t-test
if len(speed_rv1) > 1 and len(speed_rv0) > 1:
    tstat, pval, dfree = ttest_ind(speed_rv1, speed_rv0, usevar='unequal')
else:
    tstat, pval, dfree = np.nan, np.nan, np.nan

# Regression with controls using log(speed) to reduce skew
sub = sub.replace([np.inf, -np.inf], np.nan)
sub = sub[sub['speed'] > 0].copy()
sub['log_speed'] = np.log(sub['speed'])

# Select covariates
covariates = ['reader_view', 'page_id', 'num_words', 'device', 'age', 'education', 'gender',
              'english_native', 'Flesch_Kincaid', 'correct_rate', 'retake_trial']

# Keep rows with needed columns
model_df = sub[covariates + ['log_speed']].dropna().copy()

# One-hot encode categoricals
cat_cols = ['page_id', 'device', 'education', 'english_native']
model_df = pd.get_dummies(model_df, columns=cat_cols, drop_first=True)

X = model_df.drop(columns=['log_speed'])
X = sm.add_constant(X)
y = model_df['log_speed']

model = sm.OLS(y, X).fit()

# Extract reader_view coefficient
rv_coef = model.params.get('reader_view', np.nan)
rv_pval = model.pvalues.get('reader_view', np.nan)

# Print results for inspection
print('Group stats (dyslexia_bin==1):')
print(group_stats)
print('\nWelch t-test (speed, reader_view=1 vs 0):')
print(f't={tstat:.4f}, p={pval:.6f}, df={dfree:.2f}')

print('\nRegression on log(speed) with controls:')
print(f'reader_view coef={rv_coef:.6f}, p={rv_pval:.6f}')
print('\nModel summary (abbrev):')
print(model.summary().tables[1].as_text())
