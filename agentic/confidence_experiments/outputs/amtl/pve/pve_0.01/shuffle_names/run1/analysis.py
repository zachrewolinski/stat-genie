import json
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
path = 'amtl.csv'
df = pd.read_csv(path)

# Map columns based on metadata
# Outcome: number of missing teeth (AMTL) in column 'genus'
# Total observable sockets in column 'age'
# Genus (species) in column 'tooth_class'
# Tooth class in column 'sockets'
# Age at death in column 'pop'
# Sex estimate (prob male) in column 'stdev_age'

missing_raw = df['genus']
total = df['age']

# Clip missing counts to valid range to handle noise
missing = missing_raw.clip(lower=0, upper=total)

# Build analysis dataframe
adf = pd.DataFrame({
    'missing': missing,
    'total': total,
    'genus': df['tooth_class'],
    'tooth_class': df['sockets'],
    'age': df['pop'],
    'sex': df['stdev_age'],
})

# Drop any rows with non-positive totals
adf = adf[adf['total'] > 0].copy()
adf['failures'] = adf['total'] - adf['missing']

# Fit binomial GLM
formula = 'missing + failures ~ C(genus) + C(tooth_class) + age + sex'
model = smf.glm(formula=formula, data=adf, family=sm.families.Binomial()).fit()

# Average marginal predictions for each genus
mean_preds = {}
for g in adf['genus'].unique():
    tmp = adf.copy()
    tmp['genus'] = g
    pred = model.predict(tmp)
    mean_preds[g] = float(np.mean(pred))

# Extract genus coefficients and p-values (relative to reference)
params = model.params
pvalues = model.pvalues

# Identify reference genus (first in alphabetical order used by patsy)
# Patsy sets first category as reference by sorted order unless specified
# We'll reconstruct design to find baseline
cats = sorted(adf['genus'].unique())
ref_genus = cats[0]

# Collect differences for non-reference genera
genus_results = {}
for g in cats[1:]:
    term = f'C(genus)[T.{g}]'
    genus_results[g] = {
        'coef': float(params.get(term, np.nan)),
        'pvalue': float(pvalues.get(term, np.nan))
    }

# Determine evidence that Homo sapiens has higher AMTL than non-human genera
# We'll compare predicted means; also check sign/significance of coefficients if Homo is reference
homo = 'Homo sapiens'
nonhuman = [g for g in cats if g != homo]

# Determine if Homo is reference; if not, we will compare using predicted means primarily

# Summary metrics
homo_mean = mean_preds.get(homo)
nonhuman_means = {g: mean_preds[g] for g in nonhuman}

# Determine directional support
homo_higher_all = all(homo_mean > m for m in nonhuman_means.values()) if homo_mean is not None else False

# Determine significance if Homo is reference (else mark as unknown)
all_sig = None
if ref_genus == homo:
    # Homo is reference, coefficients for other genera should be negative if they have lower AMTL
    sig_flags = []
    for g in nonhuman:
        term = f'C(genus)[T.{g}]'
        coef = params.get(term, np.nan)
        pval = pvalues.get(term, np.nan)
        sig_flags.append((coef < 0) and (pval < 0.05))
    all_sig = all(sig_flags)

# Decide Likert response
# Base response on predicted differences and significance
if homo_mean is None:
    response = 50
    conclusion = 'Insufficient data to compute genus-specific predictions.'
else:
    if homo_higher_all and (all_sig is True):
        response = 85
    elif homo_higher_all:
        response = 70
    else:
        response = 35

# Build explanation
lines = []
lines.append('Model: binomial GLM of missing teeth (AMTL) with total observable sockets as trials, predictors include genus, tooth class, age at death, and sex estimate.')
lines.append('AMTL counts were clipped to the valid range [0, total sockets] to handle noise-induced negatives/overcounts.')
lines.append(f'Reference genus used by the model: {ref_genus}.')
lines.append('Average predicted AMTL rates (marginal over covariates):')
for g, m in mean_preds.items():
    lines.append(f'  - {g}: {m:.4f}')

if ref_genus == homo:
    lines.append('Genus coefficients vs Homo sapiens (negative means lower AMTL than humans):')
    for g in nonhuman:
        term = f'C(genus)[T.{g}]'
        coef = params.get(term, np.nan)
        pval = pvalues.get(term, np.nan)
        lines.append(f'  - {g}: coef={coef:.3f}, p={pval:.3g}')
else:
    lines.append('Homo sapiens is not the reference category; inference relies on marginal predictions rather than direct coefficients.')

if homo_higher_all:
    lines.append('Homo sapiens shows higher predicted AMTL rates than each non-human genus after adjusting for covariates.')
else:
    lines.append('Homo sapiens does not show consistently higher predicted AMTL rates than each non-human genus after adjustment.')

if all_sig is True:
    lines.append('The differences vs each non-human genus are statistically significant (p < 0.05).')
elif all_sig is False:
    lines.append('At least one difference vs non-human genera is not statistically significant at p < 0.05.')
else:
    lines.append('Statistical significance of genus differences was not assessed via reference-category coefficients.')

explanation = '\n'.join(lines)

# Write conclusion file
out = {
    'response': int(response),
    'explanation': explanation
}
with open('conclusion.txt', 'w') as f:
    json.dump(out, f)
