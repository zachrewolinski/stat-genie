import pandas as pd
import numpy as np
import statsmodels.formula.api as smf


df = pd.read_csv('amtl.csv')

# Set categorical with explicit reference
ref = 'Homo sapiens'

# Basic model with sockets as covariate (controls for observable sockets)
formula = 'num_amtl ~ C(genus, Treatment(reference=\"%s\")) + age + prob_male + C(tooth_class) + sockets' % ref
model = smf.ols(formula, data=df).fit(cov_type='cluster', cov_kwds={'groups': df['specimen']})

print(model.summary())

# Extract coefficients for non-human genera
coef = model.params.filter(like='C(genus')
se = model.bse.filter(like='C(genus')
pvals = model.pvalues.filter(like='C(genus')

print('\nGenus effects vs Homo sapiens (negative means lower AMTL than Homo):')
for name in coef.index:
    print(f"{name}: coef={coef[name]:.3f}, SE={se[name]:.3f}, p={pvals[name]:.4g}")

# Create a combined non-human indicator and test difference

df['is_human'] = (df['genus'] == ref).astype(int)
formula2 = 'num_amtl ~ is_human + age + prob_male + C(tooth_class) + sockets'
model2 = smf.ols(formula2, data=df).fit(cov_type='cluster', cov_kwds={'groups': df['specimen']})

print('\nHuman vs non-human model:')
print(model2.summary().tables[1])

