import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats

# Load data

df = pd.read_csv('crofoot.csv')

# Define predictors
# Relative group size
size_diff = df['n_focal'] - df['n_other']
size_ratio = df['n_focal'] / df['n_other']

# Contest location advantage: positive if focal is closer to its home-range center than the other group
loc_diff = df['dist_other'] - df['dist_focal']

# Prepare data for logistic regression
X = pd.DataFrame({
    'size_diff': size_diff,
    'loc_diff': loc_diff
})
X = sm.add_constant(X)
y = df['win']

model = sm.Logit(y, X).fit(disp=False)

# Alternative model with size_ratio to check robustness
X_ratio = pd.DataFrame({
    'size_ratio': size_ratio,
    'loc_diff': loc_diff
})
X_ratio = sm.add_constant(X_ratio)
model_ratio = sm.Logit(y, X_ratio).fit(disp=False)

# Compute odds ratios and confidence intervals
params = model.params
conf = model.conf_int()
conf.columns = ['2.5%', '97.5%']

or_params = np.exp(params)
or_conf = np.exp(conf)

# McFadden pseudo R^2
ll_null = sm.Logit(y, sm.add_constant(pd.DataFrame({'intercept': np.ones(len(y))}))).fit(disp=False).llf
ll_model = model.llf
pseudo_r2 = 1 - ll_model / ll_null

# Predicted change: compute marginal effect at mean
margeff = model.get_margeff(at='mean').summary_frame()

# Summaries
print('N:', len(df))
print('Win rate:', df['win'].mean())
print('\nMain model (size_diff + loc_diff)')
print(model.summary())
print('\nOdds ratios:')
print(or_params)
print('\nOR 95% CI:')
print(or_conf)
print('\nPseudo R^2:', pseudo_r2)
print('\nMarginal effects at mean:')
print(margeff)

print('\nRobustness model (size_ratio + loc_diff)')
print(model_ratio.summary())

