import pandas as pd
import statsmodels.api as sm


df = pd.read_csv('caschools.csv')

# Student-teacher ratio
# Avoid division by zero (should not exist, but guard anyway)
df = df.copy()
df['str'] = df['students'] / df['teachers']

# Academic performance: average of reading and math scores
# (Both are on same scale; averaging is standard)
df['avg_score'] = (df['read'] + df['math']) / 2

# Basic correlation
corr = df[['str', 'avg_score']].corr().iloc[0, 1]

# Simple regression
X1 = sm.add_constant(df['str'])
model1 = sm.OLS(df['avg_score'], X1).fit()

# Regression with controls (demographics and income)
controls = ['str', 'lunch', 'english', 'income', 'calworks']
X2 = sm.add_constant(df[controls])
model2 = sm.OLS(df['avg_score'], X2).fit()

print('Correlation (str vs avg_score):', corr)
print('\nSimple OLS: avg_score ~ str')
print(model1.summary())
print('\nControlled OLS: avg_score ~ str + lunch + english + income + calworks')
print(model2.summary())

# Save key results for conclusion
results = {
    'corr': corr,
    'simple_coef': model1.params['str'],
    'simple_pval': model1.pvalues['str'],
    'ctrl_coef': model2.params['str'],
    'ctrl_pval': model2.pvalues['str'],
}

pd.Series(results).to_csv('analysis_results.csv')
