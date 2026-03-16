import pandas as pd
import numpy as np
import statsmodels.api as sm

# Load data
df = pd.read_csv('crofoot.csv')

# Compute predictors
df['rel_size'] = df['n_focal'] - df['n_other']
df['rel_location'] = df['dist_other'] - df['dist_focal']  # positive means contest closer to focal center

# Standardize predictors for comparability
df['rel_size_z'] = (df['rel_size'] - df['rel_size'].mean()) / df['rel_size'].std(ddof=0)
df['rel_location_z'] = (df['rel_location'] - df['rel_location'].mean()) / df['rel_location'].std(ddof=0)

# Fit logistic regression
X = df[['rel_size_z', 'rel_location_z']]
X = sm.add_constant(X)
y = df['win']
model = sm.Logit(y, X)
result = model.fit(disp=False)

# Also fit single predictor models for context
X_size = sm.add_constant(df[['rel_size_z']])
res_size = sm.Logit(y, X_size).fit(disp=False)

X_loc = sm.add_constant(df[['rel_location_z']])
res_loc = sm.Logit(y, X_loc).fit(disp=False)

# Compute odds ratios and CI
def odds_ratio_ci(res):
    params = res.params
    conf = res.conf_int()
    or_vals = np.exp(params)
    or_low = np.exp(conf[0])
    or_high = np.exp(conf[1])
    return pd.DataFrame({'OR': or_vals, 'CI_low': or_low, 'CI_high': or_high, 'p': res.pvalues})

summary_main = odds_ratio_ci(result)
summary_size = odds_ratio_ci(res_size)
summary_loc = odds_ratio_ci(res_loc)

# Compute classification accuracy baseline and model
pred = (result.predict(X) >= 0.5).astype(int)
accuracy = (pred == y).mean()
baseline = max(y.mean(), 1 - y.mean())

# Pseudo R2 (McFadden)
llf = result.llf
llnull = result.llnull
mcfadden_r2 = 1 - llf/llnull

# Save results
print('n_rows', len(df))
print('win_rate', y.mean())
print('\nMain model OR + p-values:')
print(summary_main)
print('\nRel size only OR + p-values:')
print(summary_size)
print('\nRel location only OR + p-values:')
print(summary_loc)
print('\nAccuracy', accuracy, 'Baseline', baseline)
print('McFadden R2', mcfadden_r2)
