import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data

df = pd.read_csv('amtl.csv')

# Basic cleaning

df = df[df['sockets'] > 0].copy()

df['amtl_rate'] = df['num_amtl'] / df['sockets']

# Ensure categorical types and consistent ordering

genus_order = ['Pan', 'Pongo', 'Papio', 'Homo sapiens']
if set(genus_order).issubset(set(df['genus'].unique())):
    df['genus'] = pd.Categorical(df['genus'], categories=genus_order)
else:
    df['genus'] = df['genus'].astype('category')

df['tooth_class'] = df['tooth_class'].astype('category')

# Fit binomial GLM with frequency weights (number of sockets)

model = smf.glm(
    formula='amtl_rate ~ C(genus) + age + prob_male + C(tooth_class)',
    data=df,
    family=sm.families.Binomial(),
    freq_weights=df['sockets']
)
res = model.fit()

print(res.summary())

# Contrast tests: Homo sapiens vs each non-human genus

param_names = res.params.index.tolist()

# Identify baseline genus
baseline = None
for g in genus_order:
    if f"C(genus)[T.{g}]" not in param_names:
        baseline = g
        break
print(f"Baseline genus: {baseline}")

# Function to build contrast vector for comparing Homo vs another genus

def contrast_homo_vs(other):
    # We want (Homo - other) on log-odds scale
    # If baseline is one of them, adjust accordingly.
    vec = np.zeros(len(param_names))
    idx = {name: i for i, name in enumerate(param_names)}

    def add_term(genus, sign):
        term = f"C(genus)[T.{genus}]"
        if term in idx:
            vec[idx[term]] += sign
        else:
            # genus is baseline -> no explicit term
            pass

    # Homo coefficient minus other coefficient
    add_term('Homo sapiens', 1)
    add_term(other, -1)
    return vec

non_humans = ['Pan', 'Pongo', 'Papio']
for other in non_humans:
    if other == 'Homo sapiens':
        continue
    c = contrast_homo_vs(other)
    test = res.t_test(c)
    est = float(np.asarray(test.effect).item())
    se = float(np.asarray(test.sd).item())
    pval = float(np.asarray(test.pvalue).item())
    print(f"Homo vs {other}: log-odds diff={est:.4f}, SE={se:.4f}, p={pval:.4g}")

# Marginal predicted AMTL rate for each genus (standardized over observed covariates)

preds = {}
for g in df['genus'].cat.categories:
    df_tmp = df.copy()
    df_tmp['genus'] = g
    # Predict probability for each row
    p = res.predict(df_tmp)
    # Weighted average by sockets to reflect exposure
    avg = np.average(p, weights=df['sockets'])
    preds[g] = avg

print("\nMarginal predicted AMTL rates (weighted by sockets):")
for g, v in preds.items():
    print(f"{g}: {v:.4f}")

# Also compute simple adjusted comparison: is_human indicator model

df['is_human'] = (df['genus'] == 'Homo sapiens').astype(int)

model2 = smf.glm(
    formula='amtl_rate ~ is_human + age + prob_male + C(tooth_class)',
    data=df,
    family=sm.families.Binomial(),
    freq_weights=df['sockets']
)
res2 = model2.fit()
print("\nHuman indicator model:")
print(res2.summary())

coef = res2.params['is_human']
se = res2.bse['is_human']
p = res2.pvalues['is_human']
print(f"is_human log-odds coef={coef:.4f}, SE={se:.4f}, p={p:.4g}")
