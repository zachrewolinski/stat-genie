import pandas as pd
import numpy as np
import statsmodels.api as sm

# Load data
df = pd.read_csv('crofoot.csv')

# Derived predictors
df['rel_size'] = df['n_focal'] - df['n_other']
df['rel_dist'] = df['dist_other'] - df['dist_focal']  # positive => focal closer to its center

# Drop missing just in case
df_model = df[['win', 'rel_size', 'rel_dist']].dropna()

X = sm.add_constant(df_model[['rel_size', 'rel_dist']])
y = df_model['win']

# Fit logistic regression
model = sm.Logit(y, X)
result = model.fit(disp=False)

# Compute odds ratios
params = result.params
conf = result.conf_int()
conf.columns = ['2.5%', '97.5%']
or_table = np.exp(pd.concat([params, conf], axis=1))
or_table.columns = ['OR', 'OR_2.5%', 'OR_97.5%']

print('N:', len(df_model))
print('Win rate:', y.mean())
print('\nLogit coefficients:')
print(result.summary2().tables[1])
print('\nOdds ratios:')
print(or_table)

# Simple effect example: predicted win probability at mean rel_dist for rel_size values
mean_rel_dist = df_model['rel_dist'].mean()
for rel_size in [-8, -4, 0, 4, 8]:
    X_pred = pd.DataFrame({'const': 1.0, 'rel_size': [rel_size], 'rel_dist': [mean_rel_dist]})
    p = result.predict(X_pred)[0]
    print(f'Predicted win prob at rel_size={rel_size}, rel_dist=mean: {p:.3f}')

# And at mean rel_size for rel_dist values
mean_rel_size = df_model['rel_size'].mean()
for rel_dist in [-300, -150, 0, 150, 300]:
    X_pred = pd.DataFrame({'const': 1.0, 'rel_size': [mean_rel_size], 'rel_dist': [rel_dist]})
    p = result.predict(X_pred)[0]
    print(f'Predicted win prob at rel_dist={rel_dist}, rel_size=mean: {p:.3f}')
