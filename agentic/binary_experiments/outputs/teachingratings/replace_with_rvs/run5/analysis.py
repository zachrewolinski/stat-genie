import pandas as pd
import statsmodels.api as sm

# Load data
_df = pd.read_csv('teachingratings.csv')

# Baseline model: eval ~ beauty
X_base = sm.add_constant(_df[['beauty']])
model_base = sm.OLS(_df['eval'], X_base).fit(cov_type='HC3')

# Expanded model with controls
cat_cols = ['minority', 'gender', 'credits', 'division', 'native', 'tenure']
X = _df[['beauty', 'age', 'students', 'allstudents']].copy()
X['age2'] = _df['age'] ** 2
X = pd.concat([X, pd.get_dummies(_df[cat_cols], drop_first=True)], axis=1)
X = sm.add_constant(X)
model_full = sm.OLS(_df['eval'], X).fit(cov_type='HC3')

print('Baseline model (eval ~ beauty)')
print(model_base.summary())
print('\nFull model (controls added)')
print(model_full.summary())

# Key coefficient for reporting
print('\nBeauty coefficient (baseline):', model_base.params['beauty'], 'p=', model_base.pvalues['beauty'])
print('Beauty coefficient (full):', model_full.params['beauty'], 'p=', model_full.pvalues['beauty'])
