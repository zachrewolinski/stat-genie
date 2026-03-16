import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm
from scipy import stats

# Load data
_df = pd.read_csv('teachingratings.csv')

# Basic correlation
beauty = _df['beauty']
ratings = _df['allstudents']

corr = stats.pearsonr(beauty, ratings)
print(f"Pearson r: {corr.statistic:.3f}, p={corr.pvalue:.4g}")

# Simple OLS
model_simple = smf.ols('allstudents ~ beauty', data=_df).fit(cov_type='HC3')
print('\nSimple OLS (HC3):')
print(model_simple.summary().tables[1])

# Prepare controls - treat categorical columns as categories
# Exclude 'division' since it's essentially an ID (unique per row)
# Use age, rownames, minority, students as numeric controls
controls = ['age', 'rownames', 'minority', 'students',
            'C(eval)', 'C(tenure)', 'C(prof)', 'C(native)', 'C(gender)', 'C(credits)']
formula = 'allstudents ~ beauty + ' + ' + '.join(controls)

model_full = smf.ols(formula, data=_df).fit(cov_type='HC3')
print('\nFull OLS with controls (HC3):')
print(model_full.summary().tables[1])

# Also compute standardized effect for beauty (beta)
# Standardize beauty and allstudents
_df_std = _df.copy()
_df_std['beauty_z'] = (_df['beauty'] - _df['beauty'].mean()) / _df['beauty'].std(ddof=0)
_df_std['allstudents_z'] = (_df['allstudents'] - _df['allstudents'].mean()) / _df['allstudents'].std(ddof=0)
model_std = smf.ols('allstudents_z ~ beauty_z', data=_df_std).fit(cov_type='HC3')
print('\nStandardized OLS (HC3):')
print(model_std.summary().tables[1])

# Mean ratings by beauty quartile for effect size intuition
_df['beauty_quartile'] = pd.qcut(_df['beauty'], 4, labels=False)
quartile_means = _df.groupby('beauty_quartile')['allstudents'].mean()
print('\nMean rating by beauty quartile (0=lowest):')
print(quartile_means)
