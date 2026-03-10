import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm

# Load data
path = 'mortgage.csv'
df = pd.read_csv(path)

print('rows', len(df))
print('columns', df.columns.tolist())

key_cols = ['female','accept','deny']
print('missing key', df[key_cols].isna().sum())

# Contingency table
ct = pd.crosstab(df['female'], df['accept'])
print('crosstab female x accept')
print(ct)

# Approval rates
rates = df.groupby('female')['accept'].mean()
print('approval rates by female', rates.to_dict())

# Chi-square test
chi2, p, dof, expected = stats.chi2_contingency(ct)
print('chi2', chi2, 'p', p)

# Difference in proportions (female - male) with CI
p_f = rates.loc[1]
p_m = rates.loc[0]
n_f = (df['female'] == 1).sum()
n_m = (df['female'] == 0).sum()
se = np.sqrt(p_f*(1-p_f)/n_f + p_m*(1-p_m)/n_m)
diff = p_f - p_m
ci_low = diff - 1.96*se
ci_high = diff + 1.96*se
print('diff female-male', diff, '95% CI', (ci_low, ci_high))

# Logistic regression: accept ~ female (unadjusted)
reg_df = df[['accept','female']].dropna()
X = sm.add_constant(reg_df['female'])
model = sm.Logit(reg_df['accept'], X).fit(disp=False)
print(model.summary())

# Adjusted model
control_cols = ['female','black','married','self_employed','mortgage_credit','consumer_credit','bad_history','housing_expense_ratio','PI_ratio','loan_to_value']
reg_df2 = df[['accept'] + control_cols].dropna()
X2 = sm.add_constant(reg_df2[control_cols])
model2 = sm.Logit(reg_df2['accept'], X2).fit(disp=False)
print(model2.summary())

coef = model2.params['female']
se2 = model2.bse['female']
or_val = np.exp(coef)
or_low = np.exp(coef - 1.96*se2)
or_high = np.exp(coef + 1.96*se2)
print('adjusted OR female', or_val, '95% CI', (or_low, or_high), 'p', model2.pvalues['female'])

coef_u = model.params['female']
se_u = model.bse['female']
or_u = np.exp(coef_u)
or_u_low = np.exp(coef_u - 1.96*se_u)
or_u_high = np.exp(coef_u + 1.96*se_u)
print('unadjusted OR female', or_u, '95% CI', (or_u_low, or_u_high), 'p', model.pvalues['female'])

