import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import norm

# Load data

df = pd.read_csv('amtl.csv')

# Basic cleaning: drop rows with missing values in needed columns
cols = ['num_amtl', 'sockets', 'age', 'prob_male', 'tooth_class', 'genus']
df = df.dropna(subset=cols).copy()

# Ensure categorical types with controlled reference level
# Use Pan as reference (non-human), so Homo coefficient is vs Pan
cat_order = ['Pan', 'Papio', 'Pongo', 'Homo sapiens']
df['genus'] = pd.Categorical(df['genus'], categories=cat_order)

# Tooth class categorical
if df['tooth_class'].dtype.name != 'category':
    df['tooth_class'] = df['tooth_class'].astype('category')

# Proportion and weights for binomial GLM
# Use freq_weights = sockets, endog = proportion missing
# (equivalent to binomial counts with weights)
df['prop_amtl'] = df['num_amtl'] / df['sockets']

model = smf.glm(
    'prop_amtl ~ C(genus) + age + prob_male + C(tooth_class)',
    data=df,
    family=sm.families.Binomial(),
    freq_weights=df['sockets']
).fit()

print(model.summary())

# Wald tests for Homo vs each non-human genus
params = model.params
cov = model.cov_params()

# Helper for linear contrast
param_names = params.index.tolist()

def wald_test(contrast):
    # contrast: dict of param->weight
    c = np.zeros(len(param_names))
    for k, v in contrast.items():
        if k not in param_names:
            raise KeyError(f"Param {k} not in model")
        c[param_names.index(k)] = v
    est = float(np.dot(c, params))
    var = float(np.dot(c, np.dot(cov.values, c)))
    se = np.sqrt(var)
    z = est / se if se > 0 else np.nan
    p = 2 * (1 - norm.cdf(abs(z)))
    return est, se, z, p

# Parameter names for genus levels
homo = 'C(genus)[T.Homo sapiens]'
papio = 'C(genus)[T.Papio]'
pongo = 'C(genus)[T.Pongo]'

results = {}

# Homo vs Pan: just homo coefficient
if homo in param_names:
    est = params[homo]
    se = np.sqrt(cov.loc[homo, homo])
    z = est / se if se > 0 else np.nan
    p = 2 * (1 - norm.cdf(abs(z)))
    results['Homo vs Pan'] = (est, se, z, p)

# Homo vs Papio: homo - papio
if homo in param_names and papio in param_names:
    results['Homo vs Papio'] = wald_test({homo: 1.0, papio: -1.0})

# Homo vs Pongo: homo - pongo
if homo in param_names and pongo in param_names:
    results['Homo vs Pongo'] = wald_test({homo: 1.0, pongo: -1.0})

print("\nWald tests (log-odds difference, positive => higher AMTL in Homo):")
for k, (est, se, z, p) in results.items():
    print(f"{k}: est={est:.4f}, se={se:.4f}, z={z:.2f}, p={p:.4g}")

# Marginal standardized predicted AMTL for each genus
# For each genus, replace genus in each row and average predicted prob
pred_means = {}
for g in cat_order:
    tmp = df.copy()
    tmp['genus'] = g
    pred = model.predict(tmp)
    pred_means[g] = float(np.average(pred, weights=tmp['sockets']))

print("\nWeighted marginal predicted AMTL proportion by genus:")
for g, v in pred_means.items():
    print(f"{g}: {v:.4f}")

# Save key results for manual conclusion if needed
with open('analysis_results.txt', 'w') as f:
    f.write(model.summary().as_text())
    f.write("\n\nWald tests (log-odds difference, positive => higher AMTL in Homo):\n")
    for k, (est, se, z, p) in results.items():
        f.write(f"{k}: est={est:.6f}, se={se:.6f}, z={z:.3f}, p={p:.6g}\n")
    f.write("\nWeighted marginal predicted AMTL proportion by genus:\n")
    for g, v in pred_means.items():
        f.write(f"{g}: {v:.6f}\n")
