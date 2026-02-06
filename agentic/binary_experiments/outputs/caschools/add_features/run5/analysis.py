import pandas as pd
import statsmodels.api as sm

# Load data
_df = pd.read_csv('caschools.csv')

# Student-teacher ratio
_df['str'] = _df['students'] / _df['teachers']

# Academic performance metrics
_df['avg_score'] = _df[['read', 'math']].mean(axis=1)

# Simple correlations
corr_read = _df['str'].corr(_df['read'])
corr_math = _df['str'].corr(_df['math'])
corr_avg = _df['str'].corr(_df['avg_score'])

print('Correlation with student-teacher ratio')
print(f"read: {corr_read:.4f}")
print(f"math: {corr_math:.4f}")
print(f"avg_score: {corr_avg:.4f}")

# Simple regression: score ~ STR
for y in ['read', 'math', 'avg_score']:
    X = sm.add_constant(_df['str'])
    model = sm.OLS(_df[y], X).fit()
    print(f"\nSimple OLS for {y}")
    print(model.summary().tables[1])

# Regression with controls commonly used in education datasets
controls = ['str', 'lunch', 'english', 'income', 'expenditure']
Xc = sm.add_constant(_df[controls])
for y in ['read', 'math', 'avg_score']:
    model = sm.OLS(_df[y], Xc).fit()
    coef = model.params['str']
    pval = model.pvalues['str']
    print(f"\nControlled OLS for {y}")
    print(f"STR coef: {coef:.4f}, p-value: {pval:.4g}")
