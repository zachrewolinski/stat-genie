import pandas as pd
import numpy as np
from statsmodels.stats.contingency_tables import Table
import statsmodels.api as sm

# Load data
_df = pd.read_csv('boxes.csv')

# Outcome labels
# y: 1=unchosen, 2=majority, 3=minority

# Age groups
bins = [3.5, 6.5, 9.5, 12.5, 14.5]
labels = ['4-6', '7-9', '10-12', '13-14']
_df['age_group'] = pd.cut(_df['age'], bins=bins, labels=labels)

# 1) Culture vs outcome (3-category) chi-square
ct_culture = pd.crosstab(_df['culture'], _df['y'])
chi2_culture = Table(ct_culture).test_nominal_association()

# 2) Age group vs outcome (3-category) chi-square
ct_age = pd.crosstab(_df['age_group'], _df['y'])
chi2_age = Table(ct_age).test_nominal_association()

# 3) Reliance on social info: demonstrated (y in {2,3}) vs unchosen (y=1)
_df['demonstrated_choice'] = (_df['y'] != 1).astype(int)
ct_culture_demo = pd.crosstab(_df['culture'], _df['demonstrated_choice'])
chi2_culture_demo = Table(ct_culture_demo).test_nominal_association()
ct_age_demo = pd.crosstab(_df['age_group'], _df['demonstrated_choice'])
chi2_age_demo = Table(ct_age_demo).test_nominal_association()

# Logistic regression: demonstrated_choice ~ age + culture
X_demo = pd.get_dummies(_df[['age', 'culture']], columns=['culture'], drop_first=True)
X_demo = sm.add_constant(X_demo)
logit_demo = sm.Logit(_df['demonstrated_choice'], X_demo).fit(disp=False)

# 4) Majority preference among demonstrated options (y in {2,3})
_demo = _df[_df['y'].isin([2, 3])].copy()
_demo['majority_choice'] = (_demo['y'] == 2).astype(int)

# Logistic regression: majority_choice ~ age + culture
X = pd.get_dummies(_demo[['age', 'culture']], columns=['culture'], drop_first=True)
X = sm.add_constant(X)
logit = sm.Logit(_demo['majority_choice'], X).fit(disp=False)

# Descriptive proportions
maj_by_culture = _demo.groupby('culture')['majority_choice'].mean().to_dict()
maj_by_age = _demo.groupby('age_group')['majority_choice'].mean().to_dict()

# Pack results
results = {
    'n_total': len(_df),
    'n_demo': len(_demo),
    'culture_outcome_chi2': float(chi2_culture.statistic),
    'culture_outcome_p': float(chi2_culture.pvalue),
    'age_outcome_chi2': float(chi2_age.statistic),
    'age_outcome_p': float(chi2_age.pvalue),
    'culture_demo_chi2': float(chi2_culture_demo.statistic),
    'culture_demo_p': float(chi2_culture_demo.pvalue),
    'age_demo_chi2': float(chi2_age_demo.statistic),
    'age_demo_p': float(chi2_age_demo.pvalue),
    'logit_demo_age_coef': float(logit_demo.params['age']),
    'logit_demo_age_p': float(logit_demo.pvalues['age']),
    'logit_demo_culture_pvalues': {k: float(v) for k, v in logit_demo.pvalues.items() if k.startswith('culture_')},
    'logit_age_coef': float(logit.params['age']),
    'logit_age_p': float(logit.pvalues['age']),
    'logit_culture_pvalues': {k: float(v) for k, v in logit.pvalues.items() if k.startswith('culture_')},
    'majority_rate_by_culture': maj_by_culture,
    'majority_rate_by_age_group': {str(k): float(v) for k, v in maj_by_age.items()},
}

pd.Series(results).to_json('analysis_results.json')

print('Culture vs outcome chi2 p:', results['culture_outcome_p'])
print('Age group vs outcome chi2 p:', results['age_outcome_p'])
print('Culture vs demonstrated chi2 p:', results['culture_demo_p'])
print('Age group vs demonstrated chi2 p:', results['age_demo_p'])
print('Logit demonstrated age p:', results['logit_demo_age_p'])
print('Logit majority age p:', results['logit_age_p'])
