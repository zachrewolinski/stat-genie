import json
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.contingency_tables import Table2x2
from scipy import stats

# Load info.json to identify columns
with open('info.json','r') as f:
    info=json.load(f)

fields=info['data_desc']['fields']

def find_column_by_desc(substr):
    for field in fields:
        desc=(field.get('properties',{}).get('description') or '').lower()
        if substr in desc:
            return field['column'], desc
    return None, None

# Identify gender column
female_col, female_desc = find_column_by_desc('female')

# Identify approval and denial columns based on direction in description
approval_col=None; approval_desc=None
denial_col=None; denial_desc=None
for field in fields:
    desc=(field.get('properties',{}).get('description') or '').lower()
    if 'accepted' in desc and 'denied' in desc and '1 if' in desc:
        if '1 if' in desc and 'accepted' in desc and 'denied' in desc:
            # Determine whether "1 if ... accepted" or "1 if ... denied"
            if '1 if' in desc and 'accepted' in desc:
                # crude direction check: look for '1 if' followed by 'accepted' before 'denied'
                idx_one = desc.find('1 if')
                idx_acc = desc.find('accepted')
                idx_den = desc.find('denied')
                if idx_one != -1 and idx_acc != -1 and idx_den != -1:
                    if idx_acc < idx_den:
                        approval_col=field['column']; approval_desc=desc
                    else:
                        denial_col=field['column']; denial_desc=desc

# Fallbacks if direction parsing failed
if approval_col is None or denial_col is None:
    for field in fields:
        desc=(field.get('properties',{}).get('description') or '').lower()
        if approval_col is None and 'accepted' in desc and 'denied' in desc:
            approval_col=field['column']; approval_desc=desc
        elif denial_col is None and 'denied' in desc and 'accepted' in desc and field['column'] != approval_col:
            denial_col=field['column']; denial_desc=desc

print('Identified columns:')
print('female_col', female_col, female_desc)
print('approval_col', approval_col, approval_desc)
print('denial_col', denial_col, denial_desc)

# Load data

df=pd.read_csv('mortgage.csv')

# Basic checks
for col in [female_col, approval_col, denial_col]:
    if col is not None:
        unique=sorted(df[col].dropna().unique())
        print(col, 'unique values', unique[:10], 'n_unique', len(unique))

# Use approval_col; if none, derive from denial (1-denial)
if approval_col is None and denial_col is not None:
    approval = 1 - df[denial_col]
    approval_col = '<derived_from_denial>'
else:
    approval = df[approval_col]

female = df[female_col]

# drop missing
mask = female.notna() & approval.notna()
sub = pd.DataFrame({'female':female[mask], 'approve':approval[mask]})

# ensure binary
print('female mean', sub['female'].mean())
print('approval mean', sub['approve'].mean())

# Contingency table
ct = pd.crosstab(sub['female'], sub['approve'])
print('contingency table:\n', ct)

# Chi-square test
chi2, p, dof, expected = stats.chi2_contingency(ct)
print('chi2', chi2, 'p', p, 'dof', dof)

# compute approval rates by gender
rates = sub.groupby('female')['approve'].mean()
print('approval rates by female:', rates.to_dict())

# Odds ratio
if ct.shape == (2,2):
    table=Table2x2(ct.values)
    print('odds ratio', table.oddsratio, 'ci', table.oddsratio_confint())

# Logistic regression (bivariate)
X = sm.add_constant(sub['female'])
model = sm.Logit(sub['approve'], X).fit(disp=False)
print(model.summary())

# Multivariate logistic regression with controls
# use all other columns as controls
controls = [c for c in df.columns if c not in {female_col, approval_col}]
# If approval derived, remove denial column if present
if denial_col in controls:
    controls.remove(denial_col)

X_full = df[[female_col] + controls].copy()
# drop columns with non-numeric? all numeric
X_full = sm.add_constant(X_full)
# align y
y_full = approval
mask_full = X_full.notna().all(axis=1) & y_full.notna()
X_full = X_full[mask_full]
y_full = y_full[mask_full]

model_full = sm.Logit(y_full, X_full).fit(disp=False, maxiter=200)
print(model_full.summary())

# Extract female coefficient/p-value
print('female coef (bivariate):', model.params['female'], 'p', model.pvalues['female'])
print('female coef (multivariate):', model_full.params['female'], 'p', model_full.pvalues['female'])
