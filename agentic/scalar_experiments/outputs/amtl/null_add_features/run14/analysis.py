import pandas as pd
import numpy as np
import statsmodels.api as sm
import patsy

# Load data
_df = pd.read_csv('amtl.csv')

cols = ['num_amtl', 'sockets', 'age', 'prob_male', 'tooth_class', 'genus']
df = _df[cols].copy()

# Clean

df = df.dropna(subset=cols)

df['genus'] = df['genus'].astype(str).str.strip()
df['tooth_class'] = df['tooth_class'].astype(str).str.strip()

df = df[(df['sockets'] > 0) & (df['num_amtl'] >= 0) & (df['num_amtl'] <= df['sockets'])].copy()

# Binary indicator for human

df['is_human'] = (df['genus'] == 'Homo sapiens').astype(int)

# Build design matrix
formula = 'is_human + age + prob_male + C(tooth_class)'
X = patsy.dmatrix(formula, df, return_type='dataframe')

# Endog as successes/failures
endog = np.column_stack([df['num_amtl'].values, (df['sockets'] - df['num_amtl']).values])

# Fit GLM binomial
model = sm.GLM(endog, X, family=sm.families.Binomial()).fit()

# Extract human effect
coef = float(model.params.get('is_human', np.nan))
se = float(model.bse.get('is_human', np.nan))
pval = float(model.pvalues.get('is_human', np.nan))

# Average marginal effect
# Predict under human=1 and human=0 using observed covariates

df_h = df.copy()
df_h['is_human'] = 1
X_h = patsy.dmatrix(formula, df_h, return_type='dataframe')


df_nh = df.copy()
df_nh['is_human'] = 0
X_nh = patsy.dmatrix(formula, df_nh, return_type='dataframe')

pred_h = model.predict(X_h)
pred_nh = model.predict(X_nh)

ame = float(np.mean(pred_h - pred_nh))

# Map to scalar -100..100

def effect_score(ame_val: float) -> float:
    a = abs(ame_val)
    if a >= 0.15:
        return 95.0
    if a >= 0.10:
        return 80.0
    if a >= 0.05:
        return 50.0
    if a >= 0.02:
        return 25.0
    if a >= 0.01:
        return 12.0
    if a >= 0.005:
        return 6.0
    return 2.0


def p_adj(p: float) -> float:
    if np.isnan(p):
        return 0.5
    if p < 0.001:
        return 1.2
    if p < 0.01:
        return 1.1
    if p < 0.05:
        return 1.0
    if p < 0.1:
        return 0.8
    if p < 0.2:
        return 0.6
    return 0.4

score = effect_score(ame) * p_adj(pval)
score = score if ame >= 0 else -score

scalar = int(np.round(np.clip(score, -100, 100)))

print('n_rows', len(df))
print('coef_is_human', coef)
print('se_is_human', se)
print('pval_is_human', pval)
print('ame', ame)
print('scalar', scalar)

with open('conclusion.txt', 'w') as f:
    f.write(str(scalar))
