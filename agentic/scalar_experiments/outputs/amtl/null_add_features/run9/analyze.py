import pandas as pd
import numpy as np
import statsmodels.api as sm
import patsy

# Load data
path = 'amtl.csv'
df = pd.read_csv(path)

# Keep relevant columns
cols = ['num_amtl','sockets','age','prob_male','tooth_class','genus']
df = df[cols].copy()

# Drop rows with missing
for c in cols:
    df = df[df[c].notna()]

# Valid rows
mask = (df['sockets'] > 0) & (df['num_amtl'] >= 0) & (df['num_amtl'] <= df['sockets'])
df = df[mask].copy()

# Binary human indicator

df['is_human'] = (df['genus'] == 'Homo sapiens').astype(int)

# Build design matrix
formula = 'is_human + age + prob_male + C(tooth_class)'

X = patsy.dmatrix(formula, df, return_type='dataframe')
y = np.column_stack([df['num_amtl'].values, (df['sockets'] - df['num_amtl']).values])

model = sm.GLM(y, X, family=sm.families.Binomial()).fit()

coef = model.params['is_human']
se = model.bse['is_human']
pval = model.pvalues['is_human']
odds_ratio = float(np.exp(coef))

# Average marginal effect by toggling is_human
X0 = X.copy()
X1 = X.copy()
if 'is_human' in X0.columns:
    X0['is_human'] = 0
    X1['is_human'] = 1

pred0 = model.predict(X0)
pred1 = model.predict(X1)
avg_diff = float(np.mean(pred1 - pred0))

with open('analysis_summary.txt', 'w') as f:
    f.write(model.summary().as_text())
    f.write('\n\n')
    f.write(f"is_human coef: {coef:.6f}\n")
    f.write(f"is_human SE: {se:.6f}\n")
    f.write(f"is_human p-value: {pval:.6g}\n")
    f.write(f"is_human odds ratio: {odds_ratio:.6f}\n")
    f.write(f"Average predicted probability diff (human - nonhuman): {avg_diff:.6f}\n")
    f.write(f"N rows: {len(df)}\n")

print('DONE')
