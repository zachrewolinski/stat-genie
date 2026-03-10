import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm

# Load data
path = 'hurricane.csv'
df = pd.read_csv(path)

# Rename columns for clarity
rename = {
    'feature1': 'id',
    'feature2': 'year',
    'feature3': 'name',
    'feature4': 'femininity_index',
    'feature5': 'min_pressure',
    'feature6': 'female_binary',
    'feature7': 'category',
    'feature8': 'deaths',
    'feature9': 'damage_2013',
    'feature10': 'years_elapsed',
    'feature11': 'source',
    'feature12': 'femininity_mturk',
    'feature13': 'max_wind',
    'feature14': 'damage_2015',
}

df = df.rename(columns=rename)

# Basic checks
print('Rows:', len(df))
print('Deaths summary:', df['deaths'].describe())
print('Femininity index summary:', df['femininity_index'].describe())

# Create transforms

df['log_deaths'] = np.log1p(df['deaths'])
df['log_damage_2015'] = np.log1p(df['damage_2015'])

# Correlations
spearman = stats.spearmanr(df['femininity_index'], df['deaths'])
pearson_log = stats.pearsonr(df['femininity_index'], df['log_deaths'])
print('Spearman femininity vs deaths:', spearman)
print('Pearson femininity vs log(deaths):', pearson_log)

# Female vs male comparison (log deaths)
log_deaths_f = df.loc[df['female_binary'] == 1, 'log_deaths']
log_deaths_m = df.loc[df['female_binary'] == 0, 'log_deaths']

# t-test
_ttest = stats.ttest_ind(log_deaths_f, log_deaths_m, equal_var=False, nan_policy='omit')
print('T-test log deaths female vs male:', _ttest)

# Mann-Whitney
_mwu = stats.mannwhitneyu(log_deaths_f, log_deaths_m, alternative='two-sided')
print('Mann-Whitney log deaths female vs male:', _mwu)

# Regression with controls
# Choose controls: category, max_wind, min_pressure, log_damage_2015, year
features = ['femininity_index', 'category', 'max_wind', 'min_pressure', 'log_damage_2015', 'year']
X = df[features]
X = sm.add_constant(X)
y = df['log_deaths']

model = sm.OLS(y, X, missing='drop').fit(cov_type='HC3')
print('\nOLS (log deaths) with femininity index + controls (HC3):')
print(model.summary())

# Regression with female_binary
features_bin = ['female_binary', 'category', 'max_wind', 'min_pressure', 'log_damage_2015', 'year']
Xb = sm.add_constant(df[features_bin])
model_bin = sm.OLS(y, Xb, missing='drop').fit(cov_type='HC3')
print('\nOLS (log deaths) with female_binary + controls (HC3):')
print(model_bin.summary())

# Alternate using femininity_mturk (if available)
if 'femininity_mturk' in df.columns:
    Xm = sm.add_constant(df[['femininity_mturk', 'category', 'max_wind', 'min_pressure', 'log_damage_2015', 'year']])
    model_m = sm.OLS(y, Xm, missing='drop').fit(cov_type='HC3')
    print('\nOLS (log deaths) with femininity_mturk + controls (HC3):')
    print(model_m.summary())
