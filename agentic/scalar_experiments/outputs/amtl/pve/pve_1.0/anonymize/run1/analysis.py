import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import numpy as np
from scipy.stats import norm

# Load data
_df = pd.read_csv('amtl.csv')

# Rename for clarity
_df = _df.rename(columns={
    'feature1': 'tooth_class',
    'feature2': 'specimen_id',
    'feature3': 'amtl_rate',
    'feature4': 'observable_sockets',
    'feature5': 'age',
    'feature6': 'age_uncertainty',
    'feature7': 'sex',
    'feature8': 'genus',
    'feature9': 'region',
})

# Fit binomial GLM using AMTL proportion as response with observable sockets as weights
formula = 'amtl_rate ~ C(genus) + age + sex + C(tooth_class)'
model = smf.glm(
    formula=formula,
    data=_df,
    family=sm.families.Binomial(),
    var_weights=_df['observable_sockets']
)
res = model.fit()

print(res.summary())

# Predict marginal mean AMTL rate by genus
levels = _df['genus'].unique()

preds = {}
for g in levels:
    temp = _df.copy()
    temp['genus'] = g
    preds[g] = res.predict(temp).mean()

print('Marginal mean predicted AMTL rate by genus:')
for g, p in preds.items():
    print(g, p)

# Contrast Homo sapiens vs others on linear predictor
params = res.params
cov = res.cov_params()

categories = sorted(_df['genus'].unique())
baseline = categories[0]  # treatment coding baseline
print('Genus categories (alphabetical):', categories)
print('Baseline:', baseline)

coef_names = params.index.tolist()

def coef_for_genus(level):
    if level == baseline:
        return 0.0
    name = f"C(genus)[T.{level}]"
    return params.get(name, 0.0)


def contrast_var(level_a, level_b):
    name_a = None if level_a == baseline else f"C(genus)[T.{level_a}]"
    name_b = None if level_b == baseline else f"C(genus)[T.{level_b}]"
    vec = np.zeros(len(coef_names))
    if name_a is not None:
        vec[coef_names.index(name_a)] = 1.0
    if name_b is not None:
        vec[coef_names.index(name_b)] = -1.0
    return float(vec @ cov.values @ vec)

homo = 'Homo sapiens'
for other in [g for g in levels if g != homo]:
    diff = coef_for_genus(homo) - coef_for_genus(other)
    var = contrast_var(homo, other)
    se = np.sqrt(var) if var > 0 else np.nan
    z = diff / se if se and se > 0 else np.nan
    p = 2 * (1 - norm.cdf(abs(z))) if se and se > 0 else np.nan
    print(f'Contrast Homo sapiens vs {other}: diff={diff:.4f}, z={z:.2f}, p={p:.3g}')

