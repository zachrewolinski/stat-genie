import pandas as pd
import numpy as np
import statsmodels.api as sm

# Load data
path = 'crofoot.csv'
df = pd.read_csv(path)

# Outcome: m_focal (1 if focal wins)
# Relative group size: f_other (focal group size) - win (other group size)
# Location: m_other (focal distance from home center) - n_focal (other distance)

df['rel_size'] = df['f_other'] - df['win']
df['rel_location'] = df['m_other'] - df['n_focal']

# Add intercept
X = df[['rel_size', 'rel_location']]
X = sm.add_constant(X)
y = df['m_focal']

model = sm.Logit(y, X)
res = model.fit(disp=False)

# Also compute model with standardized predictors for effect size
X_std = df[['rel_size', 'rel_location']].astype(float)
X_std = (X_std - X_std.mean()) / X_std.std(ddof=0)
X_std = sm.add_constant(X_std)
model_std = sm.Logit(y, X_std)
res_std = model_std.fit(disp=False)

# Simple bivariate tests: logistic regression for each predictor
res_size = sm.Logit(y, sm.add_constant(df['rel_size'])).fit(disp=False)
res_loc = sm.Logit(y, sm.add_constant(df['rel_location'])).fit(disp=False)

# Summaries
print('n:', len(df))
print('Outcome mean (focal win):', y.mean())
print('\nMultivariable logit coefficients:')
print(res.summary2().tables[1])
print('\nStd coeffs:')
print(res_std.summary2().tables[1])
print('\nBivariate rel_size:')
print(res_size.summary2().tables[1])
print('\nBivariate rel_location:')
print(res_loc.summary2().tables[1])

# Compute odds ratios and 95% CI for multivariable
params = res.params
conf = res.conf_int()
OR = np.exp(params)
OR_ci = np.exp(conf)
print('\nOdds ratios (multivariable):')
print(pd.DataFrame({
    'OR': OR,
    'CI_low': OR_ci[0],
    'CI_high': OR_ci[1],
    'p': res.pvalues
}))

# Check correlation between predictors
corr = df['rel_size'].corr(df['rel_location'])
print('\nPredictor correlation:', corr)

