import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

# Load data
_df = pd.read_csv('amtl.csv')

# Basic checks
print('rows', len(_df))
print('columns', _df.columns.tolist())

# Compute rate
_df['rate'] = _df['feature3'] / _df['feature4']

# Set categorical ordering with Homo sapiens as reference
_df['feature8'] = pd.Categorical(
    _df['feature8'],
    categories=['Homo sapiens', 'Pan', 'Papio', 'Pongo'],
    ordered=False
)
_df['feature1'] = pd.Categorical(
    _df['feature1'],
    categories=['Anterior', 'Posterior', 'Premolar'],
    ordered=False
)

# Summaries
print('\nSpecies counts:')
print(_df['feature8'].value_counts(dropna=False))

print('\nRate summary:')
print(_df['rate'].describe())

# Check invalid count-like values
invalid_missing = ((_df['feature3'] < 0) | (_df['feature3'] > _df['feature4'])).sum()
print('\nInvalid missing count values (feature3 outside [0, feature4]):', int(invalid_missing))

# Weighted least squares on rate
model = smf.wls(
    'rate ~ C(feature8) + feature5 + feature7 + C(feature1)',
    data=_df,
    weights=_df['feature4']
).fit(cov_type='HC3')

print('\nWLS model summary (coef, CI):')
params = model.params
conf = model.conf_int()
for name in params.index:
    ci_low, ci_high = conf.loc[name]
    print(f'{name}: coef={params[name]:.6f}, CI=({ci_low:.6f},{ci_high:.6f}), p={model.pvalues[name]:.6g}')

# Predicted marginal mean rate by genus at average covariates
avg_age = _df['feature5'].mean()
avg_sex = _df['feature7'].mean()
# Use each tooth class equally weighted
classes = _df['feature1'].cat.categories

def predict_for(genus):
    # average over tooth classes
    preds = []
    for cls in classes:
        row = {
            'feature8': genus,
            'feature5': avg_age,
            'feature7': avg_sex,
            'feature1': cls,
            'feature4': _df['feature4'].mean()
        }
        preds.append(model.predict(pd.DataFrame([row]))[0])
    return float(np.mean(preds))

means = {g: predict_for(g) for g in _df['feature8'].cat.categories}
print('\nPredicted marginal mean rate by genus (avg covariates):')
for g, v in means.items():
    print(g, v)

# Differences vs Homo
homo = means['Homo sapiens']
print('\nDifferences vs Homo sapiens:')
for g, v in means.items():
    if g == 'Homo sapiens':
        continue
    print(g, v - homo)
