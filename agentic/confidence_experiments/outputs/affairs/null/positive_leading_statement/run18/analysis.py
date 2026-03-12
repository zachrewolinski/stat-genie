import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.formula.api import glm

# Load data
df = pd.read_csv('affairs.csv')

# Basic checks
print('Rows:', len(df))
print(df.head())

# Create binary indicator for any affair
df['any_affair'] = (df['affairs'] > 0).astype(int)

# Group summaries by children
grouped = df.groupby('children')['affairs']
print('\nMean affairs by children:')
print(grouped.mean())
print('\nMedian affairs by children:')
print(grouped.median())
print('\nProportion with any affair by children:')
print(df.groupby('children')['any_affair'].mean())

# Two-sample t-test on affairs counts
from scipy import stats
aff_yes = df.loc[df['children'] == 'yes', 'affairs']
aff_no = df.loc[df['children'] == 'no', 'affairs']
t_stat, p_val = stats.ttest_ind(aff_yes, aff_no, equal_var=False)
print('\nT-test affairs (yes vs no children): t=%.3f, p=%.4f' % (t_stat, p_val))

# Test on any_affair proportions (chi-square)
tab = pd.crosstab(df['children'], df['any_affair'])
chi2, chi_p, dof, exp = stats.chi2_contingency(tab)
print('\nChi-square any_affair ~ children: chi2=%.3f, p=%.4f' % (chi2, chi_p))
print('Contingency table:')
print(tab)

# Fit a simple logistic regression any_affair ~ children + controls
formula_logit = 'any_affair ~ C(children) + age + yearsmarried + religiousness + education + occupation + rating'
logit_model = sm.Logit.from_formula(formula_logit, data=df).fit(disp=False)
print('\nLogistic regression results (any_affair ~ children + controls):')
print(logit_model.summary())

# Fit a Poisson regression on number of affairs
formula_pois = 'affairs ~ C(children) + age + yearsmarried + religiousness + education + occupation + rating'
pois_model = glm(formula_pois, data=df, family=sm.families.Poisson()).fit()
print('\nPoisson regression results (affairs ~ children + controls):')
print(pois_model.summary())

# Extract key effect sizes
children_coef_logit = logit_model.params.get('C(children)[T.yes]', np.nan)
children_p_logit = logit_model.pvalues.get('C(children)[T.yes]', np.nan)
children_coef_pois = pois_model.params.get('C(children)[T.yes]', np.nan)
children_p_pois = pois_model.pvalues.get('C(children)[T.yes]', np.nan)
print('\nKey coefficients for children=yes (vs no):')
print('Logit coef=%.4f, p=%.4f' % (children_coef_logit, children_p_logit))
print('Poisson coef=%.4f, p=%.4f' % (children_coef_pois, children_p_pois))

# Save a small summary for downstream interpretation
summary = {
    'mean_affairs_by_children': grouped.mean().to_dict(),
    'prop_any_affair_by_children': df.groupby('children')['any_affair'].mean().to_dict(),
    't_test_p': float(p_val),
    'chi2_p': float(chi_p),
    'logit_children_coef': float(children_coef_logit),
    'logit_children_p': float(children_p_logit),
    'pois_children_coef': float(children_coef_pois),
    'pois_children_p': float(children_p_pois),
}

import json
with open('analysis_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)
