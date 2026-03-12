import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

# Load data

df = pd.read_csv('amtl.csv')

# Ensure categorical
for col in ['genus', 'tooth_class']:
    df[col] = df[col].astype('category')

# Relevel genus to make Homo sapiens baseline
if 'Homo sapiens' in df['genus'].cat.categories:
    cats = ['Homo sapiens'] + [c for c in df['genus'].cat.categories if c != 'Homo sapiens']
    df['genus'] = df['genus'].cat.reorder_categories(cats, ordered=False)

# Fit OLS with cluster-robust SE by specimen
model = smf.ols('num_amtl ~ C(genus) + age + prob_male + C(tooth_class)', data=df).fit(
    cov_type='cluster', cov_kwds={'groups': df['specimen']}
)

print(model.summary())

# Extract coefficients for genus vs Homo
print('\nGenus coefficients vs Homo sapiens (other - Homo):')
for term in model.params.index:
    if term.startswith('C(genus)'):
        coef = model.params[term]
        se = model.bse[term]
        pval = model.pvalues[term]
        print(term, {'coef': coef, 'se': se, 'p': pval})

# Marginal means: set genus for all rows and average predictions
marginal_means = {}
for genus in df['genus'].cat.categories:
    df_temp = df.copy()
    df_temp['genus'] = genus
    pred = model.predict(df_temp)
    marginal_means[genus] = float(pred.mean())

print('\nMarginal mean num_amtl by genus (averaged over covariate distribution):')
for g, v in marginal_means.items():
    print(g, v)

# Differences Homo - other genera (positive means Homo higher)
homo_mean = marginal_means.get('Homo sapiens')
print('\nDifferences (Homo sapiens - other genera) using marginal means:')
for g, v in marginal_means.items():
    if g == 'Homo sapiens':
        continue
    print(g, homo_mean - v)
