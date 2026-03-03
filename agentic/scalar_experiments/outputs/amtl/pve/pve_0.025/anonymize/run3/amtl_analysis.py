import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

# Load data
csv_path = 'amtl.csv'

df = pd.read_csv(csv_path)

# Set categorical variables with Homo sapiens as reference

df['feature1'] = df['feature1'].astype('category')
df['feature8'] = pd.Categorical(
    df['feature8'],
    categories=['Homo sapiens', 'Pan', 'Papio', 'Pongo'],
    ordered=False,
)

# OLS model with robust standard errors
model = smf.ols('feature3 ~ C(feature8) + feature5 + feature7 + C(feature1)', data=df).fit(cov_type='HC3')

# Extract coefficients for genus comparisons
coefs = model.params
pvals = model.pvalues
conf = model.conf_int()

# Build a compact results table
rows = []
for genus in ['Pan', 'Papio', 'Pongo']:
    term = f'C(feature8)[T.{genus}]'
    rows.append({
        'comparison': f'{genus} vs Homo sapiens',
        'coef': coefs[term],
        'p_value': pvals[term],
        'ci_low': conf.loc[term, 0],
        'ci_high': conf.loc[term, 1],
    })

result_df = pd.DataFrame(rows)
print(result_df.to_string(index=False))

# Compute adjusted means by genus via marginal predictions
# Use overall covariate distribution, swap genus for all rows
adj_means = {}
for genus in ['Homo sapiens', 'Pan', 'Papio', 'Pongo']:
    df_tmp = df.copy()
    df_tmp['feature8'] = genus
    adj_means[genus] = model.predict(df_tmp).mean()

print('\nAdjusted mean feature3 by genus (model-based):')
for genus, mean in adj_means.items():
    print(f'{genus}: {mean:.4f}')

# Differences in adjusted means vs Homo
print('\nAdjusted mean differences vs Homo sapiens:')
for genus in ['Pan', 'Papio', 'Pongo']:
    diff = adj_means[genus] - adj_means['Homo sapiens']
    print(f'{genus} - Homo sapiens: {diff:.4f}')

