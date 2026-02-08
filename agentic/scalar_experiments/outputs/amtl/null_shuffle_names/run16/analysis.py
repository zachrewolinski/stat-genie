import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
_df = pd.read_csv('amtl.csv')

# Rename columns to meaningful names based on metadata inspection
_df = _df.rename(columns={
    'sockets': 'tooth_class',
    'tooth_class': 'genus',
    'genus': 'num_amtl',
    'age': 'num_sockets',
    'pop': 'age_at_death',
    'num_amtl': 'stdev_age',
    'stdev_age': 'prob_male',
    'prob_male': 'specimen_id',
    'specimen': 'population'
})

# Ensure numeric types
for col in ['num_amtl', 'num_sockets', 'age_at_death', 'prob_male', 'stdev_age']:
    _df[col] = pd.to_numeric(_df[col], errors='coerce')

# Drop rows with missing or invalid values
_df = _df.dropna(subset=['num_amtl', 'num_sockets', 'age_at_death', 'prob_male', 'tooth_class', 'genus'])

# Ensure bounds: num_amtl between 0 and num_sockets
_df = _df[(_df['num_amtl'] >= 0) & (_df['num_sockets'] > 0) & (_df['num_amtl'] <= _df['num_sockets'])]

# Build binomial model using proportion with weights
_df['amtl_prop'] = _df['num_amtl'] / _df['num_sockets']

# Set reference genus to non-human (Pan) for interpretability
_df['genus'] = pd.Categorical(_df['genus'], categories=['Pan', 'Pongo', 'Papio', 'Homo sapiens'], ordered=False)

_model = smf.glm(
    formula='amtl_prop ~ C(genus) + age_at_death + prob_male + C(tooth_class)',
    data=_df,
    family=sm.families.Binomial(),
    freq_weights=_df['num_sockets']
).fit()

# Extract coefficient for Homo sapiens (vs Pan)
coef = _model.params.get('C(genus)[T.Homo sapiens]', np.nan)
pval = _model.pvalues.get('C(genus)[T.Homo sapiens]', np.nan)

# Compute adjusted marginal effect: predict with genus set to Homo sapiens vs non-human average

def predict_with_genus(genus_value):
    tmp = _df.copy()
    tmp['genus'] = genus_value
    return _model.predict(tmp)

pred_hs = predict_with_genus('Homo sapiens')

# Average predictions over non-human genera weighted by their sample sizes
non_human = _df[_df['genus'] != 'Homo sapiens']
weights = non_human['num_sockets'].values
pred_non = _model.predict(non_human)

avg_hs = np.average(pred_hs, weights=_df['num_sockets'])
avg_non = np.average(pred_non, weights=weights)

diff = avg_hs - avg_non

print('coef_hs_vs_pan', coef)
print('pval_hs_vs_pan', pval)
print('avg_pred_hs', avg_hs)
print('avg_pred_non', avg_non)
print('diff', diff)

with open('analysis_summary.txt', 'w') as f:
    f.write(_model.summary().as_text())
