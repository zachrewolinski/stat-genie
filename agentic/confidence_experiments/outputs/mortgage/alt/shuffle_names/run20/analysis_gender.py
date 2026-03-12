import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.formula.api as smf

# Load data
path = 'mortgage.csv'
df = pd.read_csv(path)

# Identify gender and outcome using metadata descriptions
# Gender: column 'denied_PMI' per info.json description (1 if applicant is female)
# Outcome: column 'self_employed' per info.json description (1 if mortgage application was denied)

gender_col = 'denied_PMI'
denied_col = 'self_employed'

# Create approval indicator (1 if approved)
df['approved'] = 1 - df[denied_col]

# Crosstab and chi-square
ct = pd.crosstab(df[gender_col], df['approved'])
chi2, p, dof, exp = stats.chi2_contingency(ct)

# Approval rates by gender
rates = df.groupby(gender_col)['approved'].mean()

# Effect size: difference in approval rates
rate_diff = rates.loc[1] - rates.loc[0]

# Odds ratio from logistic regression (unadjusted)
logit1 = smf.logit('approved ~ ' + gender_col, data=df).fit(disp=False)
or_unadj = float(np.exp(logit1.params[gender_col]))
p_unadj = float(logit1.pvalues[gender_col])

# Adjusted logistic regression with other covariates (excluding outcome and redundant columns)
# We'll include all other columns except denied_col, deny (redundant), and approved
covariates = [c for c in df.columns if c not in [denied_col, 'deny', 'approved']]
# Ensure gender_col is included
if gender_col not in covariates:
    covariates.append(gender_col)

formula = 'approved ~ ' + ' + '.join(covariates)

adj_result = None
try:
    logit2 = smf.logit(formula, data=df).fit(disp=False, maxiter=200)
    adj_result = {
        'or': float(np.exp(logit2.params[gender_col])),
        'p': float(logit2.pvalues[gender_col])
    }
except Exception as e:
    adj_result = {'error': str(e)}

print('gender_col', gender_col)
print('denied_col', denied_col)
print('crosstab\n', ct)
print('chi2', chi2, 'p', p)
print('approval rates', rates.to_dict())
print('rate diff', rate_diff)
print('unadj OR', or_unadj, 'p', p_unadj)
print('adj result', adj_result)
