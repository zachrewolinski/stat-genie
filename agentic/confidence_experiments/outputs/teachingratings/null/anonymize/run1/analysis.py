import json
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

# Load data
df = pd.read_csv('teachingratings.csv')

# Rename for readability
df = df.rename(columns={
    'feature2': 'minority',
    'feature3': 'age',
    'feature4': 'gender',
    'feature5': 'single_credit',
    'feature6': 'beauty',
    'feature7': 'rating',
    'feature8': 'division',
    'feature9': 'native_english',
    'feature10': 'tenure_track',
    'feature11': 'num_raters',
    'feature12': 'num_enrolled',
    'feature13': 'instructor_id'
})

# Ensure categorical types
cat_cols = ['minority', 'gender', 'single_credit', 'division', 'native_english', 'tenure_track']
for c in cat_cols:
    df[c] = df[c].astype('category')

# Basic correlation
corr = df['beauty'].corr(df['rating'])

# Simple OLS
model_simple = smf.ols('rating ~ beauty', data=df).fit()

# OLS with controls (common in study): age, gender, minority, division, native_english, tenure_track,
# class size, participation
# We'll use num_raters and num_enrolled as continuous controls.
model_controls = smf.ols(
    'rating ~ beauty + age + gender + minority + single_credit + division + native_english + tenure_track + num_raters + num_enrolled',
    data=df
).fit()

# Effect size per 1 SD of beauty
beauty_sd = df['beauty'].std()
coef_simple = model_simple.params['beauty']
coef_controls = model_controls.params['beauty']

# Predicted rating change per SD
effect_simple_sd = coef_simple * beauty_sd
effect_controls_sd = coef_controls * beauty_sd

# p-values
p_simple = model_simple.pvalues['beauty']
p_controls = model_controls.pvalues['beauty']

results = {
    'n': int(df.shape[0]),
    'corr': corr,
    'simple_coef': coef_simple,
    'simple_p': p_simple,
    'controls_coef': coef_controls,
    'controls_p': p_controls,
    'effect_simple_sd': effect_simple_sd,
    'effect_controls_sd': effect_controls_sd,
    'rating_mean': df['rating'].mean(),
    'rating_sd': df['rating'].std(),
    'beauty_sd': beauty_sd,
}

with open('analysis_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(json.dumps(results, indent=2))
