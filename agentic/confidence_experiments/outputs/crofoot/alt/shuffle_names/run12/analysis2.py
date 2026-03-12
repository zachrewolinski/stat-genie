import pandas as pd
import numpy as np
import statsmodels.api as sm


df = pd.read_csv('crofoot.csv')

# outcome
y = df['m_focal']

# distances
d1 = df['m_other']
# other distance candidate
d2 = df['n_focal']

# evaluate simple association
indicator1 = (d1 < d2).astype(int)
indicator2 = (d2 < d1).astype(int)

print('win rate when d1<d2:', y[indicator1==1].mean(), 'n', indicator1.sum())
print('win rate when d2<d1:', y[indicator2==1].mean(), 'n', indicator2.sum())

# check both assignments for size difference
# group sizes: f_other (group A), win (group B)
size_a = df['f_other']
size_b = df['win']

# logistic with size diff and distance diff assuming A is focal and d1 is focal distance
X = pd.DataFrame({
    'size_diff': size_a - size_b,
    'dist_diff': d1 - d2
})
X = sm.add_constant(X)
model = sm.Logit(y, X).fit(disp=0)
print('Model A (A focal, d1 focal) coef:')
print(model.params)
print(model.pvalues)

# alternative: B is focal and d2 is focal distance
X2 = pd.DataFrame({
    'size_diff': size_b - size_a,
    'dist_diff': d2 - d1
})
X2 = sm.add_constant(X2)
model2 = sm.Logit(y, X2).fit(disp=0)
print('Model B (B focal, d2 focal) coef:')
print(model2.params)
print(model2.pvalues)

