import json
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from scipy import stats

# Load data
path = 'teachingratings.csv'
df = pd.read_csv(path)

# Rename columns for clarity
colmap = {
    'feature1':'course_id',
    'feature2':'minority',
    'feature3':'age',
    'feature4':'gender',
    'feature5':'single_credit',
    'feature6':'beauty',
    'feature7':'rating',
    'feature8':'division',
    'feature9':'native_english',
    'feature10':'tenure_track',
    'feature11':'n_eval',
    'feature12':'n_enrolled',
    'feature13':'instructor_id',
}

df = df.rename(columns=colmap)

# Basic stats
beauty = df['beauty']
rating = df['rating']

# Pearson correlation
corr, corr_p = stats.pearsonr(beauty, rating)

# Simple OLS
model_simple = smf.ols('rating ~ beauty', data=df).fit()

# Multivariate OLS with common controls (categoricals via C())
formula_controls = (
    'rating ~ beauty + age + C(gender) + C(minority) + C(single_credit) '
    '+ C(division) + C(native_english) + C(tenure_track) + n_eval + n_enrolled'
)
model_controls = smf.ols(formula_controls, data=df).fit()

# Extract beauty coefficients and CI
simple_coef = model_simple.params['beauty']
simple_p = model_simple.pvalues['beauty']
simple_ci = model_simple.conf_int().loc['beauty'].tolist()

ctrl_coef = model_controls.params['beauty']
ctrl_p = model_controls.pvalues['beauty']
ctrl_ci = model_controls.conf_int().loc['beauty'].tolist()

# Standardize effect: per 1 SD beauty, predicted change in rating (simple and controls)
beauty_sd = beauty.std(ddof=1)
std_effect_simple = simple_coef * beauty_sd
std_effect_ctrl = ctrl_coef * beauty_sd

results = {
    'n': int(df.shape[0]),
    'corr': float(corr),
    'corr_p': float(corr_p),
    'simple_coef': float(simple_coef),
    'simple_p': float(simple_p),
    'simple_ci_low': float(simple_ci[0]),
    'simple_ci_high': float(simple_ci[1]),
    'ctrl_coef': float(ctrl_coef),
    'ctrl_p': float(ctrl_p),
    'ctrl_ci_low': float(ctrl_ci[0]),
    'ctrl_ci_high': float(ctrl_ci[1]),
    'beauty_sd': float(beauty_sd),
    'std_effect_simple': float(std_effect_simple),
    'std_effect_ctrl': float(std_effect_ctrl),
    'rating_mean': float(rating.mean()),
    'rating_sd': float(rating.std(ddof=1)),
}

print(json.dumps(results, indent=2))
