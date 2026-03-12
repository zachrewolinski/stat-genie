import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.proportion import proportions_ztest

# Load data
path = 'mortgage.csv'
df = pd.read_csv(path)

print('Rows:', len(df))
print('Columns:', df.columns.tolist())

# Ensure expected columns
for col in ['female', 'accept', 'deny']:
    print(col, 'unique:', df[col].dropna().unique()[:10])

# Basic counts
ct = pd.crosstab(df['female'], df['accept'])
print('\nCrosstab female x accept:')
print(ct)

# Approval rates by gender
rates = df.groupby('female')['accept'].mean()
counts = df.groupby('female')['accept'].count()
print('\nApproval rates by female:')
print(rates)
print('Counts:', counts.to_dict())

# Difference in proportions test
# female=1 vs female=0
successes = [ct.loc[1, 1], ct.loc[0, 1]]
ns = [ct.loc[1].sum(), ct.loc[0].sum()]
stat, pval = proportions_ztest(successes, ns)
print('\nTwo-proportion z-test (female=1 vs female=0):')
print('z:', stat, 'p:', pval)

# Chi-square test
chi2, chi_p, dof, expected = stats.chi2_contingency(ct)
print('\nChi-square test:')
print('chi2:', chi2, 'p:', chi_p)

# Effect size (difference in proportions)
rate_f = rates.loc[1]
rate_m = rates.loc[0]
print('\nRate difference (female - male):', rate_f - rate_m)

# Logistic regression (unadjusted)
# accept ~ female
model1 = smf.logit('accept ~ female', data=df).fit(disp=False)
print('\nLogit accept ~ female:')
print(model1.summary())

# Build adjusted model with relevant mortgage variables if present
candidate_controls = [
    'black', 'housing_expense_ratio', 'self_employed', 'married',
    'mortgage_credit', 'consumer_credit', 'bad_history', 'PI_ratio',
    'loan_to_value', 'denied_PMI', 'age'
]
controls = [c for c in candidate_controls if c in df.columns]

# Remove rows with missing values in model variables
vars_for_model = ['accept', 'female'] + controls
model_df = df[vars_for_model].dropna().copy()

print('\nAdjusted model variables:', controls)
print('Rows used (adjusted):', len(model_df))

if len(controls) > 0:
    formula = 'accept ~ female + ' + ' + '.join(controls)
    model2 = smf.logit(formula, data=model_df).fit(disp=False)
    print('\nLogit accept ~ female + controls:')
    print(model2.summary())

    # Odds ratio for female
    params = model2.params
    conf = model2.conf_int()
    or_female = np.exp(params['female'])
    or_low = np.exp(conf.loc['female', 0])
    or_high = np.exp(conf.loc['female', 1])
    print('\nFemale odds ratio (adjusted):', or_female)
    print('95% CI:', (or_low, or_high))

# Also compute unadjusted odds ratio
params1 = model1.params
conf1 = model1.conf_int()
or1 = np.exp(params1['female'])
or1_low = np.exp(conf1.loc['female', 0])
or1_high = np.exp(conf1.loc['female', 1])
print('\nFemale odds ratio (unadjusted):', or1)
print('95% CI:', (or1_low, or1_high))
