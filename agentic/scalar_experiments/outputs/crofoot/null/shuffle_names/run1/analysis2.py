import pandas as pd
import numpy as np
import statsmodels.api as sm

# Load data
path = 'crofoot.csv'
df = pd.read_csv(path)

# Map columns by description from info.json
# m_focal: win indicator (1 focal wins)
# f_other: number of individuals in focal group (focal size)
# win: number of individuals in other group (other size)
# m_other: distance of focal group from center (dist_focal)
# n_focal: distance of other group from center (dist_other)

win = df['m_focal']
focal_size = df['f_other']
other_size = df['win']

# Distances
focal_dist = df['m_other']
other_dist = df['n_focal']

# Derived predictors
size_diff = focal_size - other_size
size_ratio = focal_size / other_size

dist_diff = other_dist - focal_dist  # positive means closer to focal (other further) if focal_dist smaller

# Build dataframe for modeling
model_df = pd.DataFrame({
    'win': win,
    'size_diff': size_diff,
    'size_ratio': size_ratio,
    'focal_dist': focal_dist,
    'other_dist': other_dist,
    'dist_diff': dist_diff,
})

print(model_df.describe())

# Logistic regression with size_diff and dist_diff
X = model_df[['size_diff','dist_diff']]
X = sm.add_constant(X)
model = sm.Logit(model_df['win'], X).fit(disp=False)
print('Logit win ~ size_diff + dist_diff')
print(model.summary())

# Alternative: size_ratio and dist_diff
X2 = model_df[['size_ratio','dist_diff']]
X2 = sm.add_constant(X2)
model2 = sm.Logit(model_df['win'], X2).fit(disp=False)
print('Logit win ~ size_ratio + dist_diff')
print(model2.summary())

# Univariate models
for col in ['size_diff','size_ratio','dist_diff','focal_dist','other_dist']:
    Xc = sm.add_constant(model_df[[col]])
    m = sm.Logit(model_df['win'], Xc).fit(disp=False)
    print(f'Logit win ~ {col}')
    print(m.summary())

# Simple group: mean win by size_diff sign and dist_diff sign
model_df['size_adv'] = np.where(size_diff>0,'focal larger', np.where(size_diff<0,'focal smaller','equal'))
model_df['dist_adv'] = np.where(dist_diff>0,'closer to focal', np.where(dist_diff<0,'closer to other','equal'))
print(model_df.groupby('size_adv')['win'].mean())
print(model_df.groupby('dist_adv')['win'].mean())

