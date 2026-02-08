import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data

df = pd.read_csv('amtl.csv')

# Clean: remove rows with missing key fields or nonpositive sockets
key_cols = ['num_amtl','sockets','age','prob_male','tooth_class','genus']
df = df.dropna(subset=key_cols)
df = df[df['sockets'] > 0].copy()

# Human indicator
df['is_human'] = (df['genus'] == 'Homo sapiens').astype(int)

# Ensure categories
for col in ['tooth_class']:
    df[col] = df[col].astype('category')

# Binomial response as successes/failures
endog = np.column_stack([df['num_amtl'], df['sockets'] - df['num_amtl']])

# Design matrix with intercept
X = pd.get_dummies(df[['is_human','age','prob_male','tooth_class']], drop_first=True)
X = sm.add_constant(X, has_constant='add')

# Fit GLM binomial
model = sm.GLM(endog, X, family=sm.families.Binomial())
res = model.fit()

coef = res.params['is_human']
se = res.bse['is_human']
pval = res.pvalues['is_human']

# Average marginal effect of is_human on probability
# Compute predicted probabilities with is_human=1 vs 0
X1 = X.copy()
X0 = X.copy()
X1['is_human'] = 1
X0['is_human'] = 0
p1 = res.predict(X1)
p0 = res.predict(X0)

ate = float(np.mean(p1 - p0))

# Basic scaling for Likert score
# Start with sign of ATE; magnitude based on effect size and p-value
# Effect size: |ATE| mapped to 0-60 for 0 to 0.15
# Significance: p<0.05 add 20, p<0.01 add 30

mag = min(abs(ate) / 0.15, 1.0) * 60
sig = 0
if pval < 0.01:
    sig = 30
elif pval < 0.05:
    sig = 20

score = (mag + sig)
score = score if ate > 0 else -score

# Clamp and round to int
score = int(np.round(max(min(score, 100), -100)))

# Save detailed results for reference
with open('analysis_summary.txt','w') as f:
    f.write(f"N={len(df)}\n")
    f.write(f"coef_is_human={coef}\n")
    f.write(f"se_is_human={se}\n")
    f.write(f"pval_is_human={pval}\n")
    f.write(f"ate_is_human={ate}\n")
    f.write(f"score={score}\n")

with open('conclusion.txt','w') as f:
    f.write(str(score))

print('Done')
