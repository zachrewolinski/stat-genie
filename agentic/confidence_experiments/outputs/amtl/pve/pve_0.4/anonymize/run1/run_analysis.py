import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

# Load data
_df = pd.read_csv('amtl.csv')

# Ensure categorical
_df['feature1'] = _df['feature1'].astype('category')
_df['feature8'] = _df['feature8'].astype('category')
_df['feature2'] = _df['feature2'].astype('category')

# Set reference category for genus to Homo sapiens
_df['feature8'] = _df['feature8'].cat.reorder_categories(
    ['Homo sapiens', 'Pan', 'Papio', 'Pongo'], ordered=False
)

# Fit OLS with cluster-robust SE by specimen
formula = 'feature3 ~ C(feature8, Treatment(reference="Homo sapiens")) + feature5 + feature7 + C(feature1)'
model = smf.ols(formula, data=_df).fit(cov_type='cluster', cov_kwds={'groups': _df['feature2']})

print(model.summary())

# Extract genus differences vs Homo
params = model.params
pvalues = model.pvalues

for genus in ['Pan', 'Papio', 'Pongo']:
    key = f'C(feature8, Treatment(reference="Homo sapiens"))[T.{genus}]'
    if key in params:
        print(f"{genus} vs Homo coef: {params[key]:.4f}, p={pvalues[key]:.4g}")

# Adjusted means by setting genus and averaging predictions
adj_means = {}
for genus in ['Homo sapiens', 'Pan', 'Papio', 'Pongo']:
    df_tmp = _df.copy()
    df_tmp['feature8'] = genus
    adj_means[genus] = model.predict(df_tmp).mean()

print('Adjusted mean (feature3) by genus:')
for k,v in adj_means.items():
    print(k, v)

# Differences Homo - others
for genus in ['Pan', 'Papio', 'Pongo']:
    diff = adj_means['Homo sapiens'] - adj_means[genus]
    print(f"Adj mean diff Homo - {genus}: {diff:.4f}")

