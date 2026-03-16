import json
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

# Load data
path = 'amtl.csv'
df = pd.read_csv(path)

# Keep relevant columns
cols = ['num_amtl','age','prob_male','tooth_class','genus']
df = df[cols].copy()

# Drop missing
for c in cols:
    df = df[df[c].notna()]

# Homo indicator
df['homo'] = (df['genus'] == 'Homo sapiens').astype(int)

# Ensure categorical
df['tooth_class'] = df['tooth_class'].astype('category')

# Basic group stats
mean_by_genus = df.groupby('genus')['num_amtl'].agg(['mean','std','count']).reset_index()
mean_homo = df.loc[df['homo']==1,'num_amtl'].mean()
mean_non = df.loc[df['homo']==0,'num_amtl'].mean()

# Cohen's d (Hedges g) for Homo vs non-human
n1 = df['homo'].sum()
n0 = (df['homo']==0).sum()
var1 = df.loc[df['homo']==1,'num_amtl'].var(ddof=1)
var0 = df.loc[df['homo']==0,'num_amtl'].var(ddof=1)
pooled = ((n1-1)*var1 + (n0-1)*var0) / (n1+n0-2)
cohen_d = (mean_homo - mean_non) / np.sqrt(pooled) if pooled>0 else np.nan
# Hedges g correction
J = 1 - (3/(4*(n1+n0)-9))
hedges_g = cohen_d * J if not np.isnan(cohen_d) else np.nan

# Regression with controls
model = smf.ols('num_amtl ~ homo + age + prob_male + C(tooth_class)', data=df).fit(cov_type='HC3')

coef = model.params.get('homo', np.nan)
pval = model.pvalues.get('homo', np.nan)

# Save stats for inspection
out = {
    'n': int(len(df)),
    'mean_by_genus': mean_by_genus.to_dict(orient='records'),
    'mean_homo': float(mean_homo),
    'mean_non': float(mean_non),
    'hedges_g': float(hedges_g),
    'coef_homo': float(coef),
    'pval_homo': float(pval)
}

print(json.dumps(out, indent=2))
