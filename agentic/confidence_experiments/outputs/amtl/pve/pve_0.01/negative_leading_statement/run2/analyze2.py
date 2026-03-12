import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

# Load data

df = pd.read_csv('amtl.csv')
for col in ['genus','tooth_class','specimen']:
    df[col] = df[col].astype('category')

# Create human indicator

df['is_human'] = (df['genus'] == 'Homo sapiens').astype(int)

# OLS with genus dummies (Homo sapiens as reference)
model = smf.ols('num_amtl ~ C(genus) + age + prob_male + C(tooth_class)', data=df).fit()

# Cluster-robust SE by specimen
model_cluster = model.get_robustcov_results(cov_type='cluster', groups=df['specimen'])

# OLS with is_human
model_human = smf.ols('num_amtl ~ is_human + age + prob_male + C(tooth_class)', data=df).fit()
model_human_cluster = model_human.get_robustcov_results(cov_type='cluster', groups=df['specimen'])

print('=== OLS genus (non-robust) ===')
print(model.summary())
print('\n=== OLS genus (cluster by specimen) ===')
print(model_cluster.summary())

print('\n=== OLS human indicator (non-robust) ===')
print(model_human.summary())
print('\n=== OLS human indicator (cluster by specimen) ===')
print(model_human_cluster.summary())

# Extract coefficients for genus and human indicator

def extract_coef(result, name):
    idx = list(result.params.index).index(name)
    coef = result.params[idx]
    se = result.bse[idx]
    t = result.tvalues[idx]
    p = result.pvalues[idx]
    return coef, se, t, p

# For cluster results, use names
print('\nCoefficients (cluster)')
for term in ['C(genus)[T.Pan]', 'C(genus)[T.Papio]', 'C(genus)[T.Pongo]']:
    if term in model_cluster.params.index:
        coef, se, t, p = extract_coef(model_cluster, term)
        print(term, coef, se, t, p)

if 'is_human' in model_human_cluster.params.index:
    coef, se, t, p = extract_coef(model_human_cluster, 'is_human')
    print('is_human', coef, se, t, p)

