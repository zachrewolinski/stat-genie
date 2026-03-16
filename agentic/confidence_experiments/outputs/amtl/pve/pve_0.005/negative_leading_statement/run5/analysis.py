import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

# Load data
_df = pd.read_csv('amtl.csv')

# Compute rate (noisy proportion)
_df['amtl_rate'] = _df['num_amtl'] / _df['sockets']

# Ensure categorical ordering
_df['genus'] = pd.Categorical(_df['genus'], categories=['Homo sapiens', 'Pan', 'Papio', 'Pongo'])
_df['tooth_class'] = pd.Categorical(_df['tooth_class'], categories=['Anterior', 'Premolar', 'Posterior'])

# Fit linear model with robust SE
model = smf.ols('amtl_rate ~ C(genus, Treatment(reference="Homo sapiens")) + age + prob_male + C(tooth_class, Treatment(reference="Anterior"))', data=_df).fit()
robust = model.get_robustcov_results(cov_type='HC3')

# Extract genus effects with labeled indices
param_index = model.params.index
params = pd.Series(robust.params, index=param_index)
pvalues = pd.Series(robust.pvalues, index=param_index)

# Build summary for genus comparisons
results = []
for g in ['Pan', 'Papio', 'Pongo']:
    term = f'C(genus, Treatment(reference="Homo sapiens"))[T.{g}]'
    results.append({
        'genus': g,
        'coef_vs_homo': params.get(term, np.nan),
        'pvalue': pvalues.get(term, np.nan)
    })

# Estimated marginal means by genus at mean covariates
mean_age = _df['age'].mean()
mean_male = _df['prob_male'].mean()
# Use observed tooth_class distribution to average predictions
pred_rows = []
for g in ['Homo sapiens', 'Pan', 'Papio', 'Pongo']:
    for tc, weight in _df['tooth_class'].value_counts(normalize=True).items():
        pred_rows.append({'genus': g, 'age': mean_age, 'prob_male': mean_male, 'tooth_class': tc, 'weight': weight})
_pred_df = pd.DataFrame(pred_rows)
_pred_df['pred'] = model.predict(_pred_df)
mean_preds = _pred_df.groupby('genus').apply(lambda d: np.average(d['pred'], weights=d['weight']))

# Save outputs to a small json for inspection
import json
out = {
    'n': len(_df),
    'mean_rate': float(_df['amtl_rate'].mean()),
    'genus_effects': results,
    'mean_preds': {k: float(v) for k, v in mean_preds.items()},
    'model_r2': float(model.rsquared),
}

with open('analysis_results.json', 'w') as f:
    json.dump(out, f, indent=2)

print(json.dumps(out, indent=2))
