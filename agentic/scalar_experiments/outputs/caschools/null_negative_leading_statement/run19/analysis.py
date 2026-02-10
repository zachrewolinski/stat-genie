import pandas as pd
import statsmodels.api as sm

# Load data
df = pd.read_csv('caschools.csv')

# Construct student-teacher ratio
# teachers is FTE count, students is total enrollment
# Ratio = students per teacher: higher means larger classes

df['stratio'] = df['students'] / df['teachers']

# Outcome variables: reading, math, and their average
df['avgscore'] = df[['read', 'math']].mean(axis=1)

# Simple correlations
corr_read = df['stratio'].corr(df['read'])
corr_math = df['stratio'].corr(df['math'])
corr_avg = df['stratio'].corr(df['avgscore'])

print('Correlation(stratio, read):', corr_read)
print('Correlation(stratio, math):', corr_math)
print('Correlation(stratio, avgscore):', corr_avg)

# Bivariate OLS: avgscore on stratio
X_simple = sm.add_constant(df['stratio'])
model_simple = sm.OLS(df['avgscore'], X_simple).fit()
print('\nSimple OLS avgscore ~ stratio')
print(model_simple.summary())

# Multiple regression controlling for key covariates that affect achievement
controls = ['income', 'english', 'lunch', 'calworks', 'expenditure']
X_multi = sm.add_constant(df[['stratio'] + controls])
model_multi = sm.OLS(df['avgscore'], X_multi).fit()
print('\nMultiple OLS avgscore ~ stratio + controls')
print(model_multi.summary())

# Summarize direction and strength of association
coef_simple = model_simple.params['stratio']
coef_multi = model_multi.params['stratio']

print('\ncoef_simple (stratio):', coef_simple)
print('coef_multi (stratio):', coef_multi)

# Heuristic mapping to Likert scalar (-100 to 100)
# We interpret more negative coefficients (since larger ratio is worse) as evidence that
# lower ratio is associated with higher performance.

# Use multiple regression coefficient as primary; fall back to simple if needed
coef = coef_multi

# If coefficient is positive or very close to zero, that supports "No".
# If clearly negative and robust, that supports "Yes".

if coef >= 0:
    scalar = -80  # clear evidence against lower ratio helping
else:
    # Scale by magnitude relative to score SD (~20) and typical range of stratio.
    stratio_range = (df['stratio'].max() - df['stratio'].min())
    effect_range = coef * stratio_range  # change in avgscore across range
    # Map effect size in score points to scalar
    if abs(effect_range) < 1:
        scalar = -40  # essentially negligible effect despite negative sign
    elif abs(effect_range) < 3:
        scalar = -10  # small effect, weak evidence
    else:
        scalar = 20   # moderate effect contrary to prior belief

print('\nSelected scalar:', scalar)

with open('conclusion.txt', 'w') as f:
    f.write(str(int(round(scalar))))
