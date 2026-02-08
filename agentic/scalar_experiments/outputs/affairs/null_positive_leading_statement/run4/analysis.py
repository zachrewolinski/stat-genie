import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.formula.api as smf
import statsmodels.api as sm

# Load data
path = 'affairs.csv'
df = pd.read_csv(path)

# Encode children
# children: yes/no

# Basic summary
summary = {}
summary['n'] = len(df)
summary['children_counts'] = df['children'].value_counts().to_dict()

# Mean affairs by children
means = df.groupby('children')['affairs'].mean()
counts = df.groupby('children')['affairs'].size()
stds = df.groupby('children')['affairs'].std()

# t-test difference in means (unequal variances)
no_vals = df.loc[df['children'] == 'no', 'affairs']
yes_vals = df.loc[df['children'] == 'yes', 'affairs']
ttest = stats.ttest_ind(yes_vals, no_vals, equal_var=False, nan_policy='omit')

# Any affair (binary)
df['any_affair'] = (df['affairs'] > 0).astype(int)
any_rates = df.groupby('children')['any_affair'].mean()

# Logistic regression (any affair) with controls
logit_formula = 'any_affair ~ C(children) + C(gender) + age + yearsmarried + religiousness + education + occupation + rating'
logit_model = smf.logit(logit_formula, data=df).fit(disp=False)

# OLS for counts (affairs) with controls
ols_formula = 'affairs ~ C(children) + C(gender) + age + yearsmarried + religiousness + education + occupation + rating'
ols_model = smf.ols(ols_formula, data=df).fit()

# Poisson regression for counts
poisson_model = smf.glm(ols_formula, data=df, family=sm.families.Poisson()).fit()

# Extract children effect (yes vs no baseline)
# By default, C(children)[T.yes] means yes compared to no baseline (alphabetical). confirm
# Determine baseline category
children_cats = sorted(df['children'].dropna().unique())

# Extract coefficients
logit_coef = logit_model.params.get('C(children)[T.yes]', np.nan)
logit_p = logit_model.pvalues.get('C(children)[T.yes]', np.nan)

ols_coef = ols_model.params.get('C(children)[T.yes]', np.nan)
ols_p = ols_model.pvalues.get('C(children)[T.yes]', np.nan)

poisson_coef = poisson_model.params.get('C(children)[T.yes]', np.nan)
poisson_p = poisson_model.pvalues.get('C(children)[T.yes]', np.nan)

# Convert Poisson coef to incidence rate ratio
poisson_irr = np.exp(poisson_coef) if np.isfinite(poisson_coef) else np.nan

# Logistic odds ratio
logit_or = np.exp(logit_coef) if np.isfinite(logit_coef) else np.nan

# Print results
print('N:', summary['n'])
print('Children counts:', summary['children_counts'])
print('Mean affairs by children:')
print(means)
print('Std affairs by children:')
print(stds)
print('Any affair rate by children:')
print(any_rates)
print('T-test (yes vs no) for affairs mean:')
print(ttest)
print('\nLogit (any affair) coef for children yes:', logit_coef, 'p=', logit_p, 'OR=', logit_or)
print('OLS (affairs) coef for children yes:', ols_coef, 'p=', ols_p)
print('Poisson (affairs) coef for children yes:', poisson_coef, 'p=', poisson_p, 'IRR=', poisson_irr)

# Also compute predicted differences from models
# Predicted mean affairs for yes/no at observed covariates (average marginal prediction)

# Create copies with children set to yes/no
base = df.copy()
base_yes = base.copy()
base_no = base.copy()
base_yes['children'] = 'yes'
base_no['children'] = 'no'

# OLS predicted mean difference
ols_pred_yes = ols_model.predict(base_yes).mean()
ols_pred_no = ols_model.predict(base_no).mean()

# Poisson predicted mean difference
pois_pred_yes = poisson_model.predict(base_yes).mean()
pois_pred_no = poisson_model.predict(base_no).mean()

# Logit predicted probabilities
logit_pred_yes = logit_model.predict(base_yes).mean()
logit_pred_no = logit_model.predict(base_no).mean()

print('\nPredicted (OLS) mean affairs yes/no:', ols_pred_yes, ols_pred_no, 'diff=', ols_pred_yes-ols_pred_no)
print('Predicted (Poisson) mean affairs yes/no:', pois_pred_yes, pois_pred_no, 'diff=', pois_pred_yes-pois_pred_no)
print('Predicted (Logit) any affair prob yes/no:', logit_pred_yes, logit_pred_no, 'diff=', logit_pred_yes-logit_pred_no)
