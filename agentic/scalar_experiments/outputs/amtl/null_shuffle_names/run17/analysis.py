import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import norm

# Load data
_df = pd.read_csv('amtl.csv')

# Map shuffled columns to semantic names
# Based on value patterns in info.json vs data
_df = _df.rename(columns={
    'sockets': 'tooth_class',
    'prob_male': 'specimen_id',
    'genus': 'num_amtl',
    'age': 'num_sockets',
    'pop': 'age_at_death',
    'num_amtl': 'stdev_age',
    'stdev_age': 'prob_male',
    'tooth_class': 'genus',
    'specimen': 'population',
})

# Basic cleaning
_df = _df.copy()
_df['num_amtl'] = pd.to_numeric(_df['num_amtl'], errors='coerce')
_df['num_sockets'] = pd.to_numeric(_df['num_sockets'], errors='coerce')
_df['age_at_death'] = pd.to_numeric(_df['age_at_death'], errors='coerce')
_df['prob_male'] = pd.to_numeric(_df['prob_male'], errors='coerce')

# Drop rows with missing essentials
_df = _df.dropna(subset=['num_amtl', 'num_sockets', 'age_at_death', 'prob_male', 'tooth_class', 'genus'])

# Ensure counts are valid
_df = _df[_df['num_sockets'] > 0]
_df = _df[_df['num_amtl'] >= 0]
_df = _df[_df['num_amtl'] <= _df['num_sockets']]

# Set categorical ordering with Homo sapiens as reference
_df['genus'] = pd.Categorical(_df['genus'], categories=[
    'Homo sapiens', 'Pan', 'Papio', 'Pongo'
])
_df = _df.dropna(subset=['genus'])

# Binomial GLM using proportion with weights
_df['amtl_rate'] = _df['num_amtl'] / _df['num_sockets']

formula = (
    'amtl_rate ~ C(genus, Treatment(reference="Homo sapiens")) '
    '+ age_at_death + prob_male + C(tooth_class)'
)

model = smf.glm(
    formula=formula,
    data=_df,
    family=sm.families.Binomial(),
    freq_weights=_df['num_sockets']
).fit()

# Compute contrasts: Homo sapiens vs each non-human genus (log-odds scale)
params = model.params
cov = model.cov_params()

non_human = ['Pan', 'Papio', 'Pongo']
contrasts = []

for g in non_human:
    term = f'C(genus, Treatment(reference="Homo sapiens"))[T.{g}]'
    if term in params.index:
        # coefficient is (non-human - Homo). Homo - nonhuman = -coef
        coef = params[term]
        se = np.sqrt(cov.loc[term, term])
        z = -coef / se if se > 0 else np.nan
        p = 2 * (1 - norm.cdf(abs(z))) if np.isfinite(z) else np.nan
        contrasts.append({'genus': g, 'coef_nonhuman_minus_homo': coef, 'z_homo_gt': z, 'p': p})

# Predicted marginal rates by genus at mean covariates, averaged over tooth classes
mean_age = _df['age_at_death'].mean()
mean_prob_male = _df['prob_male'].mean()

# Weight by observed tooth_class distribution
class_weights = _df['tooth_class'].value_counts(normalize=True)

rates = {}
for g in ['Homo sapiens'] + non_human:
    preds = []
    weights = []
    for cls, w in class_weights.items():
        row = {
            'genus': g,
            'age_at_death': mean_age,
            'prob_male': mean_prob_male,
            'tooth_class': cls,
        }
        pred = model.predict(pd.DataFrame([row]))[0]
        preds.append(pred)
        weights.append(w)
    rates[g] = float(np.average(preds, weights=weights))

homo_rate = rates['Homo sapiens']
nonhuman_rate = float(np.mean([rates[g] for g in non_human]))
rate_diff = homo_rate - nonhuman_rate

# Aggregate evidence
if contrasts:
    mean_logodds_diff = float(np.mean([-c['coef_nonhuman_minus_homo'] for c in contrasts]))
    mean_p = float(np.mean([c['p'] for c in contrasts]))
else:
    mean_logodds_diff = 0.0
    mean_p = 1.0

# Map to Likert scale (-100..100)
# Use tanh on log-odds difference for effect size, dampened by (1 - mean_p)
raw = np.tanh(mean_logodds_diff) * (1 - mean_p)
scalar = int(np.clip(round(raw * 100), -100, 100))

# If the rate difference is opposite sign to log-odds, align with rate difference
if rate_diff < 0 and scalar > 0:
    scalar = -scalar
elif rate_diff > 0 and scalar < 0:
    scalar = -scalar

with open('conclusion.txt', 'w') as f:
    f.write(str(int(scalar)))

print('Homo rate:', homo_rate)
print('Nonhuman mean rate:', nonhuman_rate)
print('Rate diff:', rate_diff)
print('Mean log-odds diff (Homo - nonhuman):', mean_logodds_diff)
print('Mean p:', mean_p)
print('Scalar:', scalar)
