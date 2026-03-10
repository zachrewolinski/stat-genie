import pandas as pd
import statsmodels.api as sm

_df = pd.read_csv('mortgage.csv')

# Map columns based on descriptions in info.json
# gender: column described as "1 if applicant is female, 0 if male"
# outcome: column described as "1 if mortgage application was denied, 0 if accepted"

gender_col = 'denied_PMI'
outcome_col = 'self_employed'

# Basic rates
print('gender_col mean (female rate):', _df[gender_col].mean())
print('outcome_col mean (denial rate):', _df[outcome_col].mean())

# Crosstab
ct = pd.crosstab(_df[gender_col], _df[outcome_col], normalize='index')
print('\nCrosstab (row proportions)')
print(ct)

rate_f = _df.loc[_df[gender_col]==1, outcome_col].mean()
rate_m = _df.loc[_df[gender_col]==0, outcome_col].mean()
print('denial rate female', rate_f, 'male', rate_m, 'diff (female-male)', rate_f-rate_m)

# Logistic regression: denial ~ female
y = _df[outcome_col]
X = sm.add_constant(_df[gender_col])
model = sm.Logit(y, X, missing='drop').fit(disp=False)
print('\nLogit denial ~ female')
print(model.summary())

# Add controls (numeric columns excluding gender/outcome)
controls = [c for c in _df.columns if c not in [gender_col, outcome_col]]
controls = [c for c in controls if pd.api.types.is_numeric_dtype(_df[c])]
controls = [c for c in controls if _df[c].isna().mean() < 0.5]
controls = [c for c in controls if _df[c].nunique(dropna=True) > 1]
controls = [c for c in controls if _df[c].nunique(dropna=True) / len(_df) < 0.9]

X2 = _df[[gender_col] + controls].copy()
X2 = sm.add_constant(X2)
try:
    model2 = sm.Logit(y, X2, missing='drop').fit(disp=False)
    print('\nLogit denial ~ female + controls')
    print('controls', controls)
    print(model2.summary())
except Exception as e:
    print('Error fitting controlled model', e)
