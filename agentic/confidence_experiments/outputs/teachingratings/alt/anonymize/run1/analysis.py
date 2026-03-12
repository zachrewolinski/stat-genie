import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from scipy import stats

# Load data
path = 'teachingratings.csv'
df = pd.read_csv(path)

# Rename columns for clarity
rename = {
    'feature1': 'course_id',
    'feature2': 'minority',
    'feature3': 'age',
    'feature4': 'gender',
    'feature5': 'single_credit',
    'feature6': 'beauty',
    'feature7': 'rating',
    'feature8': 'division',
    'feature9': 'native_english',
    'feature10': 'tenure_track',
    'feature11': 'num_participants',
    'feature12': 'num_enrolled',
    'feature13': 'instructor_id',
}

df = df.rename(columns=rename)

# Basic stats
n = len(df)
beauty_mean = df['beauty'].mean()
rating_mean = df['rating'].mean()

# Correlation
corr = df['beauty'].corr(df['rating'])
# Pearson correlation test
corr_r, corr_p = stats.pearsonr(df['beauty'], df['rating'])

# Simple linear regression
model_simple = smf.ols('rating ~ beauty', data=df).fit()

# Multiple regression with controls
# Use categorical variables as C()
model_full = smf.ols(
    'rating ~ beauty + age + C(gender) + C(minority) + C(single_credit) + '
    'C(division) + C(native_english) + C(tenure_track) + num_participants + num_enrolled',
    data=df
).fit()

# Extract relevant results
simple_coef = model_simple.params['beauty']
simple_p = model_simple.pvalues['beauty']

full_coef = model_full.params['beauty']
full_p = model_full.pvalues['beauty']

# Standardized effect (beta) for beauty in simple regression
beauty_std = df['beauty'].std(ddof=1)
rating_std = df['rating'].std(ddof=1)
std_beta_simple = simple_coef * (beauty_std / rating_std)

# For full model, compute standardized coefficient via z-scoring
zdf = df.copy()
for col in ['beauty', 'age', 'num_participants', 'num_enrolled']:
    zdf[col] = (zdf[col] - zdf[col].mean()) / zdf[col].std(ddof=1)

model_full_std = smf.ols(
    'rating ~ beauty + age + C(gender) + C(minority) + C(single_credit) + '
    'C(division) + C(native_english) + C(tenure_track) + num_participants + num_enrolled',
    data=zdf
).fit()

std_beta_full = model_full_std.params['beauty']

# 95% CI for beauty effect
simple_ci = model_simple.conf_int().loc['beauty'].tolist()
full_ci = model_full.conf_int().loc['beauty'].tolist()

# Print outputs
print('N', n)
print('beauty_mean', beauty_mean)
print('rating_mean', rating_mean)
print('corr', corr, 'p', corr_p)
print('simple_coef', simple_coef, 'p', simple_p, 'ci', simple_ci, 'std_beta', std_beta_simple, 'r2', model_simple.rsquared)
print('full_coef', full_coef, 'p', full_p, 'ci', full_ci, 'std_beta', std_beta_full, 'r2', model_full.rsquared)

# Additional check: robust SEs (HC3)
model_full_hc3 = model_full.get_robustcov_results(cov_type='HC3')
full_coef_hc3 = model_full_hc3.params[model_full.model.exog_names.index('beauty')]
full_p_hc3 = model_full_hc3.pvalues[model_full.model.exog_names.index('beauty')]
print('full_coef_hc3', full_coef_hc3, 'p_hc3', full_p_hc3)
