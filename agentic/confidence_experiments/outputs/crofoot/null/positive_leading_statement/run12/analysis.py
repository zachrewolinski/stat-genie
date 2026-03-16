import pandas as pd
import statsmodels.api as sm
import numpy as np

# Load data

df = pd.read_csv('crofoot.csv')

# Create predictors

df['rel_size'] = df['n_focal'] - df['n_other']
# Positive means contest is closer to focal home-range center

df['rel_dist'] = df['dist_other'] - df['dist_focal']

# Prepare design matrix

X = df[['rel_size', 'rel_dist']].copy()
X = sm.add_constant(X)
y = df['win']

# Fit logistic regression

model = sm.Logit(y, X)
result = model.fit(disp=False)

# Extract stats

params = result.params
pvalues = result.pvalues
conf = result.conf_int()

# Compute odds ratios and CI

or_vals = np.exp(params)
or_ci = np.exp(conf)

# Simple bivariate models for context

bivariate = {}
for var in ['rel_size', 'rel_dist']:
    Xb = sm.add_constant(df[[var]])
    rb = sm.Logit(y, Xb).fit(disp=False)
    bivariate[var] = {
        'coef': float(rb.params[var]),
        'pvalue': float(rb.pvalues[var]),
        'or': float(np.exp(rb.params[var]))
    }

# Predicted probabilities for illustrative contrasts

size_min, size_max = df['rel_size'].min(), df['rel_size'].max()
dist_min, dist_max = df['rel_dist'].min(), df['rel_dist'].max()


def predict_prob(rel_size, rel_dist):
    Xp = sm.add_constant(
        pd.DataFrame({'rel_size': [rel_size], 'rel_dist': [rel_dist]}),
        has_constant='add'
    )
    return float(result.predict(Xp)[0])

mean_size = float(df['rel_size'].mean())
mean_dist = float(df['rel_dist'].mean())

pred_size_min = predict_prob(size_min, mean_dist)
pred_size_max = predict_prob(size_max, mean_dist)
pred_dist_min = predict_prob(mean_size, dist_min)
pred_dist_max = predict_prob(mean_size, dist_max)

# Print results for summary
print('N', len(df))
print('Rel_size range', size_min, size_max)
print('Rel_dist range', dist_min, dist_max)
print('\nMultivariate logistic regression: win ~ rel_size + rel_dist')
print(result.summary())
print('\nOdds ratios:')
print(or_vals)
print('\nOR CI:')
print(or_ci)
print('\nP-values:')
print(pvalues)
print('\nBivariate models:')
print(bivariate)
print('\nPredicted probabilities:')
print('rel_size min->max (rel_dist mean):', pred_size_min, '->', pred_size_max)
print('rel_dist min->max (rel_size mean):', pred_dist_min, '->', pred_dist_max)
