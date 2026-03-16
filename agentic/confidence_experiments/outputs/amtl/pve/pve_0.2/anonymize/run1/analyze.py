import json
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
_df = pd.read_csv('amtl.csv')

# Basic cleaning
needed_cols = ['feature1','feature3','feature4','feature5','feature7','feature8']
df = _df[needed_cols].copy()

# Drop rows with missing or non-positive denominators
# Keep only rows with valid socket counts
_df_before = len(df)
df = df.dropna()
df = df[df['feature4'] > 0]

# Response as proportion
# feature3 is missing count (may be non-integer). Use proportion with weights.
df['amtl_prop'] = df['feature3'] / df['feature4']

# Ensure bounded between 0 and 1 for binomial; drop any out-of-range
# Allow small numerical issues
mask = (df['amtl_prop'] >= -1e-8) & (df['amtl_prop'] <= 1 + 1e-8)
df = df[mask].copy()

# Clip to [0,1] to avoid boundary issues if any tiny deviations
_df_clip = df['amtl_prop'].clip(0, 1)
df['amtl_prop'] = _df_clip

# Fit GLM binomial with weights (denominator sockets)
formula = (
    "amtl_prop ~ C(feature8, Treatment(reference='Homo sapiens')) "
    "+ C(feature1) + feature5 + feature7"
)

model = smf.glm(
    formula=formula,
    data=df,
    family=sm.families.Binomial(),
    freq_weights=df['feature4'],
).fit()

# Collect genus coefficients
coef_table = model.summary2().tables[1].copy()

# Marginal predicted mean by genus at observed covariates (standardization)
# For each genus, set feature8 to genus and compute mean predicted probability
results = {}
for genus in df['feature8'].unique():
    tmp = df.copy()
    tmp['feature8'] = genus
    pred = model.predict(tmp)
    results[genus] = float(pred.mean())

# Differences vs Homo sapiens using coefficient and p-values
# Extract coeffs for genera
pvals = {}
coefs = {}
for genus in sorted(df['feature8'].unique()):
    if genus == 'Homo sapiens':
        continue
    term = f"C(feature8, Treatment(reference='Homo sapiens'))[T.{genus}]"
    if term in coef_table.index:
        coefs[genus] = float(coef_table.loc[term, 'Coef.'])
        pvals[genus] = float(coef_table.loc[term, 'P>|z|'])

summary = {
    'n_rows_used': int(len(df)),
    'n_rows_original': int(_df_before),
    'predicted_mean_by_genus': results,
    'genus_coefficients_logit': coefs,
    'genus_pvalues': pvals,
    'model_aic': float(model.aic),
}

print(json.dumps(summary, indent=2, sort_keys=True))
