import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from scipy import stats

# Load data
_df = pd.read_csv('mortgage.csv')

# Columns relevant to mortgage approval
cols = [
    'female', 'accept', 'deny', 'black', 'housing_expense_ratio', 'self_employed',
    'married', 'mortgage_credit', 'consumer_credit', 'bad_history', 'PI_ratio',
    'loan_to_value', 'denied_PMI'
]

# Keep only available columns (defensive)
cols = [c for c in cols if c in _df.columns]

df = _df[cols].copy()

# Drop missing values in analysis columns
before_n = len(df)
df = df.dropna()
after_n = len(df)

# Basic acceptance rates by gender
rates = df.groupby('female')['accept'].mean().rename('accept_rate')
counts = df['female'].value_counts().sort_index()

# Contingency table and chi-square test
cont_table = pd.crosstab(df['female'], df['accept'])
chi2, p_chi, dof, expected = stats.chi2_contingency(cont_table)

# Unadjusted logistic regression
model_unadj = smf.logit('accept ~ female', data=df).fit(disp=0)

# Adjusted logistic regression (controls for credit/financial factors)
formula = (
    'accept ~ female + black + housing_expense_ratio + self_employed + married '
    '+ mortgage_credit + consumer_credit + bad_history + PI_ratio + loan_to_value + denied_PMI'
)
model_adj = smf.logit(formula, data=df).fit(disp=0)

# Extract odds ratios and 95% CIs for female
import numpy as np

unadj_params = model_unadj.params
unadj_ci = model_unadj.conf_int()

adj_params = model_adj.params
adj_ci = model_adj.conf_int()

unadj_or = np.exp(unadj_params['female'])
unadj_ci_or = np.exp(unadj_ci.loc['female'])

adj_or = np.exp(adj_params['female'])
adj_ci_or = np.exp(adj_ci.loc['female'])

# Marginal effect for female in adjusted model
margeff_adj = model_adj.get_margeff(at='overall').summary_frame()
me_female = margeff_adj.loc['female'] if 'female' in margeff_adj.index else None

# Output summary stats
print('n_total', before_n)
print('n_used', after_n)
print('accept_rate_by_gender', rates.to_dict())
print('counts_by_gender', counts.to_dict())
print('contingency_table')
print(cont_table)
print('chi2', chi2, 'p', p_chi)

print('unadjusted_logit_coef_female', unadj_params['female'])
print('unadjusted_or_female', unadj_or)
print('unadjusted_or_ci_female', unadj_ci_or.to_dict())
print('unadjusted_p_female', model_unadj.pvalues['female'])

print('adjusted_logit_coef_female', adj_params['female'])
print('adjusted_or_female', adj_or)
print('adjusted_or_ci_female', adj_ci_or.to_dict())
print('adjusted_p_female', model_adj.pvalues['female'])

if me_female is not None:
    print('adjusted_marginal_effect_female', me_female.to_dict())

# Save to a CSV for debugging if needed (commented)
# df.to_csv('analysis_clean.csv', index=False)
