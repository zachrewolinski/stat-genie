import pandas as pd
import numpy as np
import statsmodels.api as sm

# Load data
df = pd.read_csv('crofoot.csv')

# Define predictors
df['size_diff'] = df['n_focal'] - df['n_other']
df['loc_adv'] = df['dist_other'] - df['dist_focal']  # positive => closer to focal home range center

# Logistic regression with both predictors
X = df[['size_diff', 'loc_adv']]
X = sm.add_constant(X)
y = df['win']

model = sm.Logit(y, X).fit(disp=False)

# Also fit univariate models for interpretability
model_size = sm.Logit(y, sm.add_constant(df[['size_diff']])).fit(disp=False)
model_loc = sm.Logit(y, sm.add_constant(df[['loc_adv']])).fit(disp=False)

# Print summaries
print('N:', len(df))
print('Win rate:', y.mean())
print('\nMultivariable logit:')
print(model.summary())
print('\nUnivariate size_diff:')
print(model_size.summary())
print('\nUnivariate loc_adv:')
print(model_loc.summary())

# Compute odds ratios and 95% CI for multivariable
params = model.params
conf = model.conf_int()
or_vals = np.exp(params)
or_ci = np.exp(conf)
out = pd.DataFrame({
    'coef': params,
    'OR': or_vals,
    'OR_ci_low': or_ci[0],
    'OR_ci_high': or_ci[1],
    'p': model.pvalues
})
print('\nOdds ratios (multivariable):')
print(out)

# Save to CSV for quick lookup if needed
out.to_csv('model_or.csv', index=True)
