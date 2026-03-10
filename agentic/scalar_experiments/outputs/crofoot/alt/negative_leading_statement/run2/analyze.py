import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
path = 'crofoot.csv'
df = pd.read_csv(path)

# create relative size and location metrics
# relative group size as difference and ratio
# difference

df['size_diff'] = df['n_focal'] - df['n_other']
# ratio (avoid division by zero not needed)
df['size_ratio'] = df['n_focal'] / df['n_other']

# location: difference in distance from home range center
# smaller distance indicates closer to home

df['dist_diff'] = df['dist_focal'] - df['dist_other']

# fit logistic regression: win ~ size_diff + dist_diff
model1 = smf.logit('win ~ size_diff + dist_diff', data=df).fit(disp=False)

# also model with size_ratio + dist_diff
model2 = smf.logit('win ~ size_ratio + dist_diff', data=df).fit(disp=False)

# model with both distances separately and size_diff
model3 = smf.logit('win ~ size_diff + dist_focal + dist_other', data=df).fit(disp=False)

# model with size_diff only
model4 = smf.logit('win ~ size_diff', data=df).fit(disp=False)

# model with dist_diff only
model5 = smf.logit('win ~ dist_diff', data=df).fit(disp=False)

# print summaries and p-values
print('n rows', len(df))
print('win mean', df['win'].mean())

for i, m in enumerate([model1, model2, model3, model4, model5], start=1):
    print('\nModel', i)
    print(m.params)
    print(m.pvalues)
    print('AIC', m.aic)

# compute effect sizes for model1
print('\nModel1 OR for size_diff, dist_diff')
print(np.exp(model1.params))

# correlation checks
print('\nCorrelation size_diff vs dist_diff', df['size_diff'].corr(df['dist_diff']))

# descriptive by win
print('\nGroup means by win')
print(df.groupby('win')[['n_focal','n_other','size_diff','dist_focal','dist_other','dist_diff']].mean())

