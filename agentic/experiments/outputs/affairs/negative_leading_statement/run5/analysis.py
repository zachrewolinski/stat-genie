import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from statsmodels.stats.weightstats import ttest_ind
from statsmodels.stats.proportion import proportions_ztest

# Load data
_df = pd.read_csv('affairs.csv')

# Basic group stats
_df['affairs_gt0'] = (_df['affairs'] > 0).astype(int)

# Mean difference in affairs by children
mean_no = _df.loc[_df['children'] == 'no', 'affairs'].mean()
mean_yes = _df.loc[_df['children'] == 'yes', 'affairs'].mean()

# t-test for mean difference
no_vals = _df.loc[_df['children'] == 'no', 'affairs']
yes_vals = _df.loc[_df['children'] == 'yes', 'affairs']
t_stat, t_p, _ = ttest_ind(no_vals, yes_vals, usevar='unequal')

# Proportion with any affair
prop_no = _df.loc[_df['children'] == 'no', 'affairs_gt0'].mean()
prop_yes = _df.loc[_df['children'] == 'yes', 'affairs_gt0'].mean()

# z-test for proportion difference
count = [_df.loc[_df['children'] == 'no', 'affairs_gt0'].sum(),
         _df.loc[_df['children'] == 'yes', 'affairs_gt0'].sum()]
obs = [_df.loc[_df['children'] == 'no', 'affairs_gt0'].count(),
       _df.loc[_df['children'] == 'yes', 'affairs_gt0'].count()]
prop_stat, prop_p = proportions_ztest(count, obs)

# Regression with controls
ols_formula = (
    'affairs ~ C(children) + C(gender) + age + yearsmarried + '
    'religiousness + education + occupation + rating'
)
ols_model = smf.ols(ols_formula, data=_df).fit(cov_type='HC1')

logit_formula = (
    'affairs_gt0 ~ C(children) + C(gender) + age + yearsmarried + '
    'religiousness + education + occupation + rating'
)
logit_model = smf.logit(logit_formula, data=_df).fit(disp=False)

# Extract key effects
ols_children_coef = ols_model.params.get('C(children)[T.yes]')
ols_children_p = ols_model.pvalues.get('C(children)[T.yes]')

logit_children_coef = logit_model.params.get('C(children)[T.yes]')
logit_children_p = logit_model.pvalues.get('C(children)[T.yes]')

# Convert logit coef to odds ratio
odds_ratio = float(np.exp(logit_children_coef))

# Print results
print('Group means (affairs):')
print(f"children=no: {mean_no:.3f}")
print(f"children=yes: {mean_yes:.3f}")
print(f"t-test (yes vs no) t={t_stat:.3f}, p={t_p:.4f}")
print()
print('Proportion with any affair (affairs>0):')
print(f"children=no: {prop_no:.3f}")
print(f"children=yes: {prop_yes:.3f}")
print(f"z-test (yes vs no) z={prop_stat:.3f}, p={prop_p:.4f}")
print()
print('OLS with controls (affairs):')
print(f"coef(children=yes)={ols_children_coef:.3f}, p={ols_children_p:.4f}")
print()
print('Logit with controls (any affair):')
print(f"coef(children=yes)={logit_children_coef:.3f}, p={logit_children_p:.4f}, odds_ratio={odds_ratio:.3f}")
