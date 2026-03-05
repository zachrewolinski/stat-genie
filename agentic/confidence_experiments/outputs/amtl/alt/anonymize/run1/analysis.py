import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
_df = pd.read_csv('amtl.csv')

# Rename for clarity
_df = _df.rename(columns={
    'feature1': 'tooth_class',
    'feature2': 'specimen_id',
    'feature3': 'missing_teeth',
    'feature4': 'observable_sockets',
    'feature5': 'age',
    'feature6': 'age_uncertainty',
    'feature7': 'sex_est',
    'feature8': 'genus',
    'feature9': 'region'
})

# Basic cleaning
# Keep rows with valid counts and covariates
_df = _df.dropna(subset=['missing_teeth', 'observable_sockets', 'age', 'sex_est', 'tooth_class', 'genus'])
_df = _df[_df['observable_sockets'] > 0]
_df = _df[_df['missing_teeth'] >= 0]
_df = _df[_df['missing_teeth'] <= _df['observable_sockets']]

# Ensure categorical types
_df['tooth_class'] = _df['tooth_class'].astype('category')
_df['genus'] = _df['genus'].astype('category')

# Build response as successes/failures
_df['present_teeth'] = _df['observable_sockets'] - _df['missing_teeth']

# GLM binomial with counts
# Baseline genus: Homo sapiens (if present). Reorder categories if needed.
if 'Homo sapiens' in list(_df['genus'].cat.categories):
    cats = list(_df['genus'].cat.categories)
    if cats[0] != 'Homo sapiens':
        cats = ['Homo sapiens'] + [c for c in cats if c != 'Homo sapiens']
        _df['genus'] = _df['genus'].cat.reorder_categories(cats, ordered=False)

formula = 'missing_teeth + present_teeth ~ C(genus) + age + sex_est + C(tooth_class)'
model = smf.glm(
    formula=formula,
    data=_df,
    family=sm.families.Binomial()
).fit()

# Extract results for genus comparisons (vs Homo sapiens baseline)
params = model.params
conf = model.conf_int()
conf.columns = ['ci_low', 'ci_high']

results = []
for genus in ['Pan', 'Pongo', 'Papio']:
    term = f'C(genus)[T.{genus}]'
    if term in params.index:
        coef = params[term]
        pval = model.pvalues[term]
        ci_low = conf.loc[term, 'ci_low']
        ci_high = conf.loc[term, 'ci_high']
        or_est = float(np.exp(coef))
        or_low = float(np.exp(ci_low))
        or_high = float(np.exp(ci_high))
        results.append({
            'genus': genus,
            'coef_log_odds_vs_homo': float(coef),
            'p_value': float(pval),
            'odds_ratio_vs_homo': or_est,
            'or_ci_low': or_low,
            'or_ci_high': or_high
        })

# Compute predicted probabilities at mean covariates for each genus and tooth class averaged
# Use overall mean age/sex and most common tooth_class (mode)
mean_age = float(_df['age'].mean())
mean_sex = float(_df['sex_est'].mean())
mode_tooth = _df['tooth_class'].mode().iloc[0]

pred_rows = []
for genus in _df['genus'].cat.categories:
    pred_rows.append({
        'genus': genus,
        'age': mean_age,
        'sex_est': mean_sex,
        'tooth_class': mode_tooth,
        'missing_teeth': 1,  # dummy for formula; will be ignored in prediction
        'present_teeth': 1
    })

pred_df = pd.DataFrame(pred_rows)
# Use model.predict to get mean probability for missing teeth (per socket)
# For binomial with counts, predict gives expected mean of missing_teeth / (missing_teeth+present_teeth)
pred_probs = model.predict(pred_df)

pred_results = {row['genus']: float(prob) for row, prob in zip(pred_rows, pred_probs)}

output = {
    'n_rows_used': int(len(_df)),
    'model_summary': model.summary().as_text(),
    'genus_comparisons_vs_homo': results,
    'predicted_missing_prob_at_mean_covariates': pred_results
}

print(json.dumps(output, indent=2))
