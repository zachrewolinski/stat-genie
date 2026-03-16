import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

# Load data
path = 'amtl.csv'
df = pd.read_csv(path)

# Fit linear model with categorical predictors
# Use Homo sapiens as reference for genus
# Use Posterior as reference for tooth_class (default alphabetical? We'll set explicitly)

# Ensure categorical types with desired reference

df['genus'] = pd.Categorical(df['genus'], categories=['Homo sapiens','Pan','Papio','Pongo'])
df['tooth_class'] = pd.Categorical(df['tooth_class'], categories=['Posterior','Anterior','Premolar'])

model = smf.ols('num_amtl ~ C(genus) + age + prob_male + C(tooth_class)', data=df).fit(cov_type='HC3')

# Extract coefficients for genus comparisons (other - Homo)
coef = model.params
pvals = model.pvalues

# Compute adjusted means by genus via average marginal predictions
adj_means = {}
for g in df['genus'].cat.categories:
    df_tmp = df.copy()
    df_tmp['genus'] = g
    preds = model.predict(df_tmp)
    adj_means[g] = preds.mean()

# Differences vs Homo
homo_mean = adj_means['Homo sapiens']
diffs = {g: adj_means[g] - homo_mean for g in adj_means if g != 'Homo sapiens'}

# Collect genus coeffs and pvals
results = []
for g in ['Pan','Papio','Pongo']:
    term = f'C(genus)[T.{g}]'
    results.append({
        'genus': g,
        'coef_other_minus_homo': coef.get(term, np.nan),
        'p_value': pvals.get(term, np.nan),
        'adj_mean': adj_means.get(g, np.nan),
        'adj_mean_minus_homo': diffs.get(g, np.nan)
    })

print('Adjusted means (num_amtl):')
for g, m in adj_means.items():
    print(f'{g}: {m:.4f}')

print('\nGenus comparisons (other - Homo):')
for r in results:
    print(f"{r['genus']}: coef={r['coef_other_minus_homo']:.4f}, p={r['p_value']:.4g}, adj_mean_diff={r['adj_mean_minus_homo']:.4f}")

print('\nModel fit:')
print(f'R2={model.rsquared:.4f}, n={int(model.nobs)}')
