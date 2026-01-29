import pandas as pd
import numpy as np
import statsmodels.api as sm
import patsy

# Load data

df = pd.read_csv('amtl.csv')

# Map columns to meanings based on values in data
# genus: count of missing teeth (AMTL)
# age: number of observable sockets (trials)
# pop: estimated age at death
# stdev_age: probability of male (0-1)
# sockets: tooth class (Anterior/Posterior/Premolar)
# tooth_class: taxonomic genus (Homo sapiens, Pan, Papio, Pongo)

# Prepare binomial endog as successes/failures
endog = np.column_stack([df['genus'], df['age'] - df['genus']])

# Build design matrix (Homo sapiens as reference)
formula = 'C(tooth_class, Treatment(reference="Homo sapiens")) + pop + stdev_age + C(sockets)'
X = patsy.dmatrix(formula, df, return_type='dataframe')

# Fit GLM
model = sm.GLM(endog, X, family=sm.families.Binomial())
res = model.fit()

# Summarize genus effects vs Homo sapiens
params = res.params
pvalues = res.pvalues
odds_ratios = np.exp(params)

rows = []
for genus in ['Pan', 'Papio', 'Pongo']:
    key = f'C(tooth_class, Treatment(reference="Homo sapiens"))[T.{genus}]'
    rows.append({
        'genus_vs_homo': genus,
        'coef': params[key],
        'odds_ratio': odds_ratios[key],
        'p_value': pvalues[key],
    })

coef_table = pd.DataFrame(rows)

# Compute adjusted predicted AMTL rate for each genus
# Use observed covariates, only swap genus (tooth_class)
unique_genera = ['Homo sapiens', 'Pan', 'Papio', 'Pongo']

adjusted_means = []
for g in unique_genera:
    df_g = df.copy()
    df_g['tooth_class'] = g
    X_g = patsy.build_design_matrices([X.design_info], df_g)[0]
    pred = res.predict(X_g)
    adjusted_means.append({'genus': g, 'adjusted_amtl_rate': float(np.mean(pred))})

adjusted_df = pd.DataFrame(adjusted_means).sort_values('adjusted_amtl_rate', ascending=False)

# Save outputs for inspection
coef_table.to_csv('genus_effects_vs_homo.csv', index=False)
adjusted_df.to_csv('adjusted_amtl_rates.csv', index=False)

# Print key results
print('Genus effects vs Homo sapiens (log-odds coefficients):')
print(coef_table.to_string(index=False))
print('\nAdjusted predicted AMTL rates (mean probability of missing tooth):')
print(adjusted_df.to_string(index=False))
