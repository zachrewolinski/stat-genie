import pandas as pd
import numpy as np
import statsmodels.api as sm


df = pd.read_csv('crofoot.csv')

# Variables
# outcome: feature4 (1 focal won)
# relative group size: difference in group size (focal - other)
# contest location: difference in distance from home range center (focal - other)
# smaller distance -> closer to home; negative difference means contest closer to focal group center.


df['rel_size'] = df['feature7'] - df['feature8']

df['rel_dist'] = df['feature5'] - df['feature6']
# also indicator if contest closer to focal group (distance smaller)

df['closer_to_focal'] = (df['feature5'] < df['feature6']).astype(int)

# logistic regression with rel_size and rel_dist
X = df[['rel_size', 'rel_dist']]
X = sm.add_constant(X)
model = sm.Logit(df['feature4'], X)
res = model.fit(disp=False)

# alternative with closer_to_focal instead of rel_dist
X2 = sm.add_constant(df[['rel_size', 'closer_to_focal']])
model2 = sm.Logit(df['feature4'], X2)
res2 = model2.fit(disp=False)

# descriptive statistics
summary = {
    'n': len(df),
    'win_rate': df['feature4'].mean(),
    'rel_size_mean': df['rel_size'].mean(),
    'rel_size_std': df['rel_size'].std(ddof=1),
    'rel_dist_mean': df['rel_dist'].mean(),
    'rel_dist_std': df['rel_dist'].std(ddof=1),
    'closer_to_focal_rate': df['closer_to_focal'].mean(),
}

# correlation for quick check
corr_rel_size = np.corrcoef(df['rel_size'], df['feature4'])[0, 1]

corr_rel_dist = np.corrcoef(df['rel_dist'], df['feature4'])[0, 1]

# compute odds ratios and p-values
params = res.params
pvalues = res.pvalues
or_vals = np.exp(params)

params2 = res2.params
pvalues2 = res2.pvalues
or_vals2 = np.exp(params2)

print('summary', summary)
print('corr_rel_size', corr_rel_size)
print('corr_rel_dist', corr_rel_dist)
print('model1_params', params)
print('model1_pvalues', pvalues)
print('model1_or', or_vals)
print('model1_aic', res.aic)
print('model2_params', params2)
print('model2_pvalues', pvalues2)
print('model2_or', or_vals2)
print('model2_aic', res2.aic)

# compute predicted probabilities for typical changes
# e.g., effect of +1 rel_size holding rel_dist at mean
mean_rel_dist = df['rel_dist'].mean()
mean_rel_size = df['rel_size'].mean()

def pred_prob(rel_size, rel_dist):
    lin = params['const'] + params['rel_size']*rel_size + params['rel_dist']*rel_dist
    return 1/(1+np.exp(-lin))

base = pred_prob(mean_rel_size, mean_rel_dist)
plus_size = pred_prob(mean_rel_size + 1, mean_rel_dist)
minus_size = pred_prob(mean_rel_size - 1, mean_rel_dist)
plus_dist = pred_prob(mean_rel_size, mean_rel_dist + 100)  # 100 m farther from focal
minus_dist = pred_prob(mean_rel_size, mean_rel_dist - 100)

print('pred_base', base)
print('pred_plus_size', plus_size)
print('pred_minus_size', minus_size)
print('pred_plus_dist', plus_dist)
print('pred_minus_dist', minus_dist)
