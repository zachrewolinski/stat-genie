import json
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

DATA_PATH = 'mortgage.csv'

df = pd.read_csv(DATA_PATH)

# Basic cleanup: ensure numeric columns
# Identify key variables
# female: 1 if female, 0 if male
# accept: 1 if accepted, 0 if denied

# Drop rows with missing in relevant columns

# Bivariate stats
biv = df[['female','accept','deny']].dropna()

# Approval rate by gender
rates = biv.groupby('female')['accept'].mean()
counts = biv.groupby('female')['accept'].agg(['count','sum'])

# Difference in proportions test (two-sided z-test)
# female=1 vs male=0
n1 = counts.loc[1,'count']
x1 = counts.loc[1,'sum']

n0 = counts.loc[0,'count']
x0 = counts.loc[0,'sum']

p1 = x1 / n1
p0 = x0 / n0
p_pool = (x1 + x0) / (n1 + n0)
se = (p_pool * (1 - p_pool) * (1/n1 + 1/n0)) ** 0.5
z = (p1 - p0) / se
pval_z = 2 * (1 - stats.norm.cdf(abs(z)))

# Logistic regression: accept ~ female + controls
# Controls: black, housing_expense_ratio, self_employed, married,
# mortgage_credit, consumer_credit, bad_history, PI_ratio, loan_to_value, denied_PMI

control_cols = [
    'black','housing_expense_ratio','self_employed','married','mortgage_credit',
    'consumer_credit','bad_history','PI_ratio','loan_to_value','denied_PMI'
]

reg_cols = ['accept','female'] + control_cols
reg_df = df[reg_cols].dropna()

formula = 'accept ~ female + ' + ' + '.join(control_cols)
logit_model = smf.logit(formula=formula, data=reg_df)
logit_res = logit_model.fit(disp=False)

female_coef = logit_res.params['female']
female_se = logit_res.bse['female']
female_p = logit_res.pvalues['female']

# Convert to odds ratio and 95% CI
or_female = np.exp(female_coef)
ci_low = np.exp(female_coef - 1.96 * female_se)
ci_high = np.exp(female_coef + 1.96 * female_se)

# Also run unadjusted logit: accept ~ female
logit_unadj = smf.logit('accept ~ female', data=biv).fit(disp=False)
coef_unadj = logit_unadj.params['female']
se_unadj = logit_unadj.bse['female']
p_unadj = logit_unadj.pvalues['female']

result = {
    'counts': {
        'female_0': {'n': int(n0), 'accepts': int(x0), 'rate': float(p0)},
        'female_1': {'n': int(n1), 'accepts': int(x1), 'rate': float(p1)},
    },
    'diff_in_props': {
        'z': float(z),
        'p_value': float(pval_z),
        'diff': float(p1 - p0)
    },
    'logit_unadjusted': {
        'coef_female': float(coef_unadj),
        'p_value': float(p_unadj),
        'odds_ratio': float(np.exp(coef_unadj))
    },
    'logit_adjusted': {
        'coef_female': float(female_coef),
        'p_value': float(female_p),
        'odds_ratio': float(or_female),
        'ci_low': float(ci_low),
        'ci_high': float(ci_high)
    },
    'n_reg': int(reg_df.shape[0])
}

print(json.dumps(result, indent=2))
