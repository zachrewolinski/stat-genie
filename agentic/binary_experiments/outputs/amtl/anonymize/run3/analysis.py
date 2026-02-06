import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api as sm

# Load data
_df = pd.read_csv('amtl.csv')
_df = _df.rename(
    columns={
        'feature1': 'tooth_class',
        'feature2': 'specimen_id',
        'feature3': 'missing',
        'feature4': 'observed',
        'feature5': 'age',
        'feature6': 'age_uncert',
        'feature7': 'sex',
        'feature8': 'genus',
        'feature9': 'region',
    }
)

# Basic cleaning
_df = _df.copy()
_df = _df[(_df['observed'] > 0) & (_df['missing'] >= 0)]
_df['prop_missing'] = _df['missing'] / _df['observed']

# Ensure categorical ordering with Homo sapiens as reference
_df['genus'] = _df['genus'].astype('category')
_df['tooth_class'] = _df['tooth_class'].astype('category')

# Fit binomial GLM with proportion response and trial weights
formula = 'prop_missing ~ C(genus, Treatment(reference="Homo sapiens")) + age + sex + C(tooth_class)'
model = smf.glm(
    formula=formula,
    data=_df,
    family=sm.families.Binomial(),
    var_weights=_df['observed']
)
result = model.fit()

# Compute average predicted probabilities for each genus (standardized over covariates)
weights = _df['observed'].to_numpy()
unique_genera = ['Homo sapiens', 'Pan', 'Pongo', 'Papio']

pred_means = {}
for g in unique_genera:
    tmp = _df.copy()
    tmp['genus'] = g
    preds = result.predict(tmp)
    pred_means[g] = np.average(preds, weights=weights)

# Extract genus coefficients and p-values
coef = result.params
pvals = result.pvalues

comparisons = {}
for g in ['Pan', 'Pongo', 'Papio']:
    term = f'C(genus, Treatment(reference="Homo sapiens"))[T.{g}]'
    if term in coef.index:
        comparisons[g] = {
            'log_odds_diff': coef[term],
            'odds_ratio': float(np.exp(coef[term])),
            'p_value': float(pvals[term]),
        }

# Decide if Homo sapiens has higher AMTL than each non-human genus
homo_mean = pred_means['Homo sapiens']
all_higher = True
all_signif = True
for g in ['Pan', 'Pongo', 'Papio']:
    if g in pred_means:
        all_higher = all_higher and (homo_mean > pred_means[g])
    if g in comparisons:
        # With Homo as reference, negative coefficient means lower AMTL in non-human
        all_signif = all_signif and (comparisons[g]['log_odds_diff'] < 0) and (comparisons[g]['p_value'] < 0.05)

print('Model fit complete.')
print('Average predicted missing proportion by genus (standardized over covariates):')
for g in unique_genera:
    print(f'  {g}: {pred_means.get(g, np.nan):.4f}')
print('Genus comparisons vs Homo sapiens (log-odds difference, odds ratio, p-value):')
for g, vals in comparisons.items():
    print(f"  {g}: log_odds_diff={vals['log_odds_diff']:.3f}, OR={vals['odds_ratio']:.3f}, p={vals['p_value']:.4g}")

print('Homo higher than all non-human (predicted means):', all_higher)
print('All non-human lower than Homo with p<0.05:', all_signif)

# Save key outputs for downstream use if needed
out = {
    'pred_means': pred_means,
    'comparisons': comparisons,
    'homo_higher_all_pred': all_higher,
    'all_nonhuman_lower_signif': all_signif,
}

pd.Series(out).to_json('analysis_results.json')
