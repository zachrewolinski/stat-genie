import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm
import numpy as np
import patsy

# Load data
_df = pd.read_csv('amtl.csv')

# Keep relevant columns
cols = ['num_amtl', 'sockets', 'age', 'prob_male', 'tooth_class', 'genus']
missing_cols = [c for c in cols if c not in _df.columns]
if missing_cols:
    raise SystemExit(f"Missing columns: {missing_cols}")

df = _df[cols].copy()

# Drop rows with missing values or invalid sockets
for c in cols:
    df = df[df[c].notna()]

df = df[df['sockets'] > 0]

df['num_amtl'] = df['num_amtl'].astype(float)
df['sockets'] = df['sockets'].astype(float)

# Ensure num_amtl within [0, sockets]
df = df[(df['num_amtl'] >= 0) & (df['num_amtl'] <= df['sockets'])]

# Binary human indicator
# Normalize genus strings
if df['genus'].dtype.name == 'category':
    df['genus'] = df['genus'].astype(str)

df['genus'] = df['genus'].astype(str).str.strip()

df['is_human'] = (df['genus'] == 'Homo sapiens').astype(int)

# Ensure tooth_class categorical
if df['tooth_class'].dtype.name != 'category':
    df['tooth_class'] = df['tooth_class'].astype('category')

# Build design matrices
formula = 'is_human + age + prob_male + C(tooth_class)'
exog = patsy.dmatrix(formula, data=df, return_type='dataframe')
endog = np.column_stack([df['num_amtl'], df['sockets'] - df['num_amtl']])

model = sm.GLM(endog, exog, family=sm.families.Binomial())
res = model.fit()

# Extract human effect
coef = res.params.get('is_human', np.nan)
se = res.bse.get('is_human', np.nan)

# Average marginal effect: set is_human=1 vs 0 per row
pred_df_h = df.copy()
pred_df_h['is_human'] = 1
pred_df_nh = df.copy()
pred_df_nh['is_human'] = 0

exog_h = patsy.dmatrix(formula, data=pred_df_h, return_type='dataframe')
exog_nh = patsy.dmatrix(formula, data=pred_df_nh, return_type='dataframe')

p_h = res.predict(exog_h)
p_nh = res.predict(exog_nh)

marginal_diff = (p_h - p_nh).mean()

# Output summary metrics
out = {
    'n_rows': int(len(df)),
    'human_share': float(df['is_human'].mean()),
    'coef_is_human': float(coef),
    'se_is_human': float(se),
    'z_is_human': float(coef / se) if se == se and se != 0 else float('nan'),
    'pvalue_is_human': float(res.pvalues.get('is_human', np.nan)),
    'avg_pred_prob_human': float(p_h.mean()),
    'avg_pred_prob_nonhuman': float(p_nh.mean()),
    'avg_marginal_diff': float(marginal_diff),
}

print(out)
