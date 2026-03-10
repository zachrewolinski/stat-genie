import pandas as pd
import numpy as np
import statsmodels.api as sm

# Load data
path = 'crofoot.csv'
df = pd.read_csv(path)

print('columns', df.columns.tolist())
print('shape', df.shape)
print(df.head())

# Keep relevant columns
needed = ['win','n_focal','n_other','dist_focal','dist_other']
missing = [c for c in needed if c not in df.columns]
print('missing', missing)

d = df[needed].copy()
# drop NA
n_before = len(d)
d = d.dropna()
print('n after dropna', len(d), 'dropped', n_before-len(d))

# create predictors
# relative size: difference and ratio
# location advantage: other distance - focal distance (positive means closer to focal)
d['rel_size_diff'] = d['n_focal'] - d['n_other']
# ratio avoid division by zero
d['rel_size_ratio'] = d['n_focal'] / d['n_other']
d['loc_adv'] = d['dist_other'] - d['dist_focal']
# normalized location advantage
d['loc_adv_norm'] = (d['dist_other'] - d['dist_focal']) / (d['dist_other'] + d['dist_focal'])

print(d[['rel_size_diff','rel_size_ratio','loc_adv','loc_adv_norm']].describe())

# Logistic regression with diff and loc_adv
X = d[['rel_size_diff','loc_adv']]
X = sm.add_constant(X)
model = sm.Logit(d['win'], X).fit(disp=False)
print('\nLogit win ~ rel_size_diff + loc_adv')
print(model.summary())

# Logistic regression with ratio and loc_adv_norm
X2 = d[['rel_size_ratio','loc_adv_norm']]
X2 = sm.add_constant(X2)
model2 = sm.Logit(d['win'], X2).fit(disp=False)
print('\nLogit win ~ rel_size_ratio + loc_adv_norm')
print(model2.summary())

# compute simple group stats: win rate by relative size sign and location advantage sign
for col, label in [('rel_size_diff','rel_size_diff'),('loc_adv','loc_adv')]:
    d[col+'_sign'] = np.where(d[col]>0, 'positive', np.where(d[col]<0, 'negative','zero'))
    print('\nWin rate by', col, 'sign')
    print(d.groupby(col+'_sign')['win'].agg(['count','mean']))

# Simple correlation (point biserial) between win and predictors
from scipy.stats import pointbiserialr
for col in ['rel_size_diff','loc_adv']:
    r,p = pointbiserialr(d['win'], d[col])
    print(f'pointbiserial {col}: r={r:.3f}, p={p:.4f}')

# Save key stats for later use
out = {
    'n': len(d),
    'mean_win': d['win'].mean(),
    'coef_diff': model.params.to_dict(),
    'pvalues_diff': model.pvalues.to_dict(),
    'coef_ratio': model2.params.to_dict(),
    'pvalues_ratio': model2.pvalues.to_dict(),
}
print('\nKey stats', out)
