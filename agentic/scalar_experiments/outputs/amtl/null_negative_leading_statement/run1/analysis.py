import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import norm

# Load data

df = pd.read_csv('amtl.csv')

# Basic cleaning: ensure sockets >= num_amtl and non-null

df = df.dropna(subset=['num_amtl', 'sockets', 'age', 'prob_male', 'tooth_class', 'genus'])
df = df[df['sockets'] >= df['num_amtl']]

# Build success/failure for binomial GLM

df['fail'] = df['sockets'] - df['num_amtl']

# Set categorical with Homo sapiens as baseline

df['genus'] = pd.Categorical(df['genus'], categories=[
    'Homo sapiens', 'Pan', 'Pongo', 'Papio'
], ordered=False)

df['tooth_class'] = pd.Categorical(df['tooth_class'])

# Fit GLM

formula = 'num_amtl + fail ~ C(genus) + age + prob_male + C(tooth_class)'
model = smf.glm(formula=formula, data=df, family=sm.families.Binomial())
result = model.fit()

print(result.summary())

# Marginal predicted probability for each genus (average over observed covariates)

def marginal_pred_for_genus(g):
    tmp = df.copy()
    tmp['genus'] = g
    pred = result.predict(tmp)
    return pred.mean()

marginals = {g: marginal_pred_for_genus(g) for g in df['genus'].cat.categories}
print('\nMarginal predicted AMTL probability by genus (avg over covariates):')
for g, p in marginals.items():
    print(f'  {g}: {p:.4f}')

# Differences vs Homo

homo = marginals['Homo sapiens']
print('\nDifferences vs Homo sapiens:')
for g in ['Pan', 'Pongo', 'Papio']:
    diff = homo - marginals[g]
    print(f'  Homo - {g}: {diff:.4f}')

# Wald tests for genus coefficients (non-human vs Homo)

params = result.params
bse = result.bse
print('\nWald tests for genus coefficients (logit scale):')
for g in ['Pan', 'Pongo', 'Papio']:
    term = f'C(genus)[T.{g}]'
    if term in params.index:
        z = params[term] / bse[term]
        p = 2 * (1 - norm.cdf(abs(z)))
        print(f'  {g} vs Homo: coef={params[term]:.4f}, z={z:.3f}, p={p:.4g}')
    else:
        print(f'  term missing for {g}')
