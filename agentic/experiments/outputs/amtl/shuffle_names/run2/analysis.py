import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api as sm
from patsy import dmatrices, build_design_matrices

# Load data
raw = pd.read_csv('amtl.csv')

# Rename columns based on metadata inspection
# sockets -> tooth class; prob_male -> specimen id; genus -> num missing;
# age -> num observable sockets; pop -> age at death; num_amtl -> age sd;
# stdev_age -> prob male; tooth_class -> genus; specimen -> population
rename_map = {
    'sockets': 'tooth_class',
    'prob_male': 'specimen_id',
    'genus': 'num_missing',
    'age': 'num_sockets',
    'pop': 'age_at_death',
    'num_amtl': 'age_sd',
    'stdev_age': 'prob_male',
    'tooth_class': 'genus',
    'specimen': 'population',
}

df = raw.rename(columns=rename_map)

# Basic sanity checks and cleaning
df = df[df['num_sockets'] > 0].copy()
df = df[df['num_missing'] >= 0].copy()
df = df[df['num_missing'] <= df['num_sockets']].copy()
df = df.dropna(subset=['num_missing', 'num_sockets', 'age_at_death', 'prob_male', 'genus', 'tooth_class']).copy()
df['num_fail'] = df['num_sockets'] - df['num_missing']

# Fit binomial GLM: missing teeth out of observable sockets
# Predictors: genus, age_at_death, prob_male, tooth_class
formula = 'num_missing + num_fail ~ C(genus) + age_at_death + prob_male + C(tooth_class)'
y, X = dmatrices(formula, data=df, return_type='dataframe')
design_info = X.design_info
model = sm.GLM(y, X, family=sm.families.Binomial())
result = model.fit()

print(result.summary())

# Marginal predicted AMTL rate by genus holding other covariates at observed values
mean_rates = {}
for g in df['genus'].unique():
    tmp = df.copy()
    tmp['genus'] = g
    X_new = build_design_matrices([design_info], tmp, return_type='dataframe')[0]
    pred = result.predict(X_new)
    mean_rates[g] = float(np.mean(pred))

mean_rates = dict(sorted(mean_rates.items(), key=lambda x: x[0]))
print('\nMean predicted AMTL proportion by genus (marginal over covariates):')
for g, v in mean_rates.items():
    print(f'  {g}: {v:.4f}')

# Pairwise differences vs Homo sapiens
homo_rate = mean_rates.get('Homo sapiens')
if homo_rate is not None:
    print('\nDifferences in mean predicted AMTL proportion (Homo sapiens - other):')
    for g, v in mean_rates.items():
        if g == 'Homo sapiens':
            continue
        print(f'  Homo sapiens - {g}: {homo_rate - v:.4f}')
