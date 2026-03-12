import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

csv_path = 'teachingratings.csv'

df = pd.read_csv(csv_path)

# rename for clarity
cols = {
    'feature2': 'minority',
    'feature3': 'age',
    'feature4': 'gender',
    'feature5': 'single_credit',
    'feature6': 'beauty',
    'feature7': 'rating',
    'feature8': 'division',
    'feature9': 'native_english',
    'feature10': 'tenure_track',
    'feature11': 'students_eval',
    'feature12': 'students_enrolled',
}

for k, v in cols.items():
    if k in df.columns:
        df = df.rename(columns={k: v})

# basic stats
n = len(df)
beauty_mean = df['beauty'].mean()
beauty_std = df['beauty'].std(ddof=1)
rating_mean = df['rating'].mean()
rating_std = df['rating'].std(ddof=1)

# correlation
corr = df['beauty'].corr(df['rating'])

# simple regression
model_simple = smf.ols('rating ~ beauty', data=df).fit()

# multiple regression with controls
# convert categorical to categorical type
for c in ['minority', 'gender', 'single_credit', 'division', 'native_english', 'tenure_track']:
    if c in df.columns:
        df[c] = df[c].astype('category')

formula = 'rating ~ beauty + age + C(gender) + C(minority) + C(single_credit) + C(division) + C(native_english) + C(tenure_track) + students_eval + students_enrolled'
model_full = smf.ols(formula, data=df).fit()

# effect size: change in rating per 1 SD beauty
beta_simple = model_simple.params['beauty']
se_simple = model_simple.bse['beauty']
p_simple = model_simple.pvalues['beauty']
ci_simple = model_simple.conf_int().loc['beauty'].tolist()

beta_full = model_full.params['beauty']
se_full = model_full.bse['beauty']
p_full = model_full.pvalues['beauty']
ci_full = model_full.conf_int().loc['beauty'].tolist()

sd_beauty = beauty_std

# standardized effect (per 1 SD beauty)
std_effect_simple = beta_simple * sd_beauty
std_effect_full = beta_full * sd_beauty

# Save summary metrics
summary = {
    'n': n,
    'beauty_mean': beauty_mean,
    'beauty_std': beauty_std,
    'rating_mean': rating_mean,
    'rating_std': rating_std,
    'corr': corr,
    'simple': {
        'beta': beta_simple,
        'se': se_simple,
        'p': p_simple,
        'ci_low': ci_simple[0],
        'ci_high': ci_simple[1],
        'r2': model_simple.rsquared,
        'std_effect': std_effect_simple,
    },
    'full': {
        'beta': beta_full,
        'se': se_full,
        'p': p_full,
        'ci_low': ci_full[0],
        'ci_high': ci_full[1],
        'r2': model_full.rsquared,
        'std_effect': std_effect_full,
    }
}

print(summary)
