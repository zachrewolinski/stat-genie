import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.contingency_tables import Table2x2

# Load data
_df = pd.read_csv('mortgage.csv')

# Basic clean-up
if 'Unnamed: 0' in _df.columns:
    _df = _df.drop(columns=['Unnamed: 0'])

# Outcome: deny (1 = denied)
# Explanatory of interest: female (1 = female)

# Drop rows with missing values in used columns
cols = [
    'deny', 'female', 'black', 'housing_expense_ratio', 'self_employed', 'married',
    'mortgage_credit', 'consumer_credit', 'bad_history', 'PI_ratio', 'loan_to_value',
    'denied_PMI'
]
_df = _df[cols].dropna()

# Descriptive denial rates by gender
rate_by_gender = _df.groupby('female')['deny'].mean()
count_by_gender = _df['female'].value_counts().sort_index()

# 2x2 table for denial vs gender
# rows: female=0,1; cols: deny=0,1
ct = pd.crosstab(_df['female'], _df['deny']).reindex(index=[0,1], columns=[0,1], fill_value=0)

# Chi-square (nominal association) using statsmodels Table2x2
# Table2x2 expects [[a,b],[c,d]] where columns are outcomes (0/1)
# Using counts: female=0 (row0), female=1 (row1)
chi2_res = Table2x2(ct.values).test_nominal_association()
chi2 = chi2_res.statistic
p_chi2 = chi2_res.pvalue
dof = chi2_res.df

# Logistic regression with controls
X = _df[[
    'female', 'black', 'housing_expense_ratio', 'self_employed', 'married',
    'mortgage_credit', 'consumer_credit', 'bad_history', 'PI_ratio', 'loan_to_value',
    'denied_PMI'
]]
X = sm.add_constant(X, has_constant='add')

y = _df['deny']

logit_model = sm.Logit(y, X).fit(disp=False)

coef_female = logit_model.params['female']
pval_female = logit_model.pvalues['female']

# Odds ratio and 95% CI for female
or_female = np.exp(coef_female)
ci = logit_model.conf_int().loc['female']
ci_or = np.exp(ci)

# Output summary for quick inspection
print('Denial rate by gender (female=0 male, female=1 female):')
print(rate_by_gender)
print('\nCounts by gender:')
print(count_by_gender)
print('\n2x2 table (rows=female 0/1, cols=deny 0/1):')
print(ct)
print(f"\nChi-square test: chi2={chi2:.4f}, dof={dof}, p={p_chi2:.4g}")
print('\nLogit coefficient for female:')
print(f"coef={coef_female:.4f}, p={pval_female:.4g}, OR={or_female:.4f}")
print(f"95% CI for OR: [{ci_or.iloc[0]:.4f}, {ci_or.iloc[1]:.4f}]")
