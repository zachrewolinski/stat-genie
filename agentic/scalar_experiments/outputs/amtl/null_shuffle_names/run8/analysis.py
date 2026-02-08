import pandas as pd
import numpy as np
import statsmodels.api as sm
from patsy import build_design_matrices, dmatrix

# Load data
raw = pd.read_csv('amtl.csv')

# Rename columns to meaningful names based on data inspection
# sockets -> tooth_class (Anterior/Posterior/Premolar)
# tooth_class -> genus_group (Homo sapiens, Pan, Pongo, Papio)
# genus -> num_missing (integer count)
# age -> num_sockets (integer count)
# pop -> age_at_death (continuous)
# stdev_age -> prob_male (0-1)
# prob_male -> specimen_id (unused)
# specimen -> population (unused)
# num_amtl -> age_uncertainty (unused)

df = raw.rename(columns={
    'sockets': 'tooth_class',
    'tooth_class': 'genus_group',
    'genus': 'num_missing',
    'age': 'num_sockets',
    'pop': 'age_at_death',
    'stdev_age': 'prob_male',
    'prob_male': 'specimen_id',
    'specimen': 'population',
    'num_amtl': 'age_uncertainty',
})

# Keep valid binomial rows
mask = (df['num_sockets'] > 0) & (df['num_missing'] >= 0) & (df['num_missing'] <= df['num_sockets'])
df = df.loc[mask].copy()

# Ensure categorical ordering

df['genus_group'] = df['genus_group'].astype('category')
df['tooth_class'] = df['tooth_class'].astype('category')

# Fit binomial GLM with counts
# Use Pan as reference group (non-human) for interpretability
# Build design matrix explicitly to avoid ratio/weight edge cases
exog = dmatrix(
    'C(genus_group, Treatment(reference="Pan")) + age_at_death + prob_male + C(tooth_class)',
    data=df,
    return_type='dataframe',
)
design_info = exog.design_info
endog = np.column_stack([df['num_missing'].values, (df['num_sockets'] - df['num_missing']).values])
model = sm.GLM(endog, exog, family=sm.families.Binomial()).fit()

# Prepare prediction function for adjusted rates

labels = list(df['genus_group'].cat.categories)
sockets = df['num_sockets'].values

def adjusted_rate_from_exog(exog, params):
    linpred = exog @ params
    p = 1 / (1 + np.exp(-linpred))
    return float(np.sum(p * sockets) / np.sum(sockets))

# Precompute design matrices for each genus label
exog_by_genus = {
    g: build_design_matrices([design_info], df.assign(genus_group=g))[0]
    for g in labels
}

# Compute adjusted rates for each genus
rate_by_genus = {g: adjusted_rate_from_exog(exog_by_genus[g], model.params.values) for g in labels}

# Weighted average for non-human genera (Pan, Pongo, Papio)
nonhuman = [g for g in labels if g != 'Homo sapiens']
weights = {g: df.loc[df['genus_group'] == g, 'num_sockets'].sum() for g in nonhuman}
wt_total = sum(weights.values())
nonhuman_rate = sum(rate_by_genus[g] * weights[g] for g in nonhuman) / wt_total
human_rate = rate_by_genus.get('Homo sapiens', np.nan)

# Uncertainty via parameter simulation
np.random.seed(0)
params = model.params.values
cov = model.cov_params().values

n_draws = 1000
try:
    draws = np.random.multivariate_normal(params, cov, size=n_draws)
except np.linalg.LinAlgError:
    # Fallback to diagonal covariance if covariance not PD
    draws = np.random.multivariate_normal(params, np.diag(np.diag(cov)), size=n_draws)

diffs = []
for b in draws:
    rates = {g: adjusted_rate_from_exog(exog_by_genus[g], b) for g in labels}
    nh_rate = sum(rates[g] * weights[g] for g in nonhuman) / wt_total
    diffs.append(rates.get('Homo sapiens', np.nan) - nh_rate)

diffs = np.array(diffs)
mean_diff = float(np.nanmean(diffs))
std_diff = float(np.nanstd(diffs, ddof=1)) if np.isfinite(diffs).any() else float('nan')
prob_positive = float(np.mean(diffs > 0)) if np.isfinite(diffs).any() else 0.5

# Map to Likert scale [-100, 100]
# Combine confidence (prob_positive) with effect size magnitude
# Scale effect size by 5 percentage points as a meaningful threshold
scale = 0.05
mag = np.tanh(abs(mean_diff) / scale) if np.isfinite(mean_diff) else 0.0
conf = 2 * prob_positive - 1  # [-1, 1]
score = int(round(100 * conf * mag))

# Save conclusion
with open('conclusion.txt', 'w') as f:
    f.write(str(score))

# Print a brief summary for inspection
print('Adjusted rates:', rate_by_genus)
print('Human rate:', human_rate)
print('Non-human rate:', nonhuman_rate)
print('Mean diff:', mean_diff, 'Std diff:', std_diff, 'P(diff>0):', prob_positive)
print('Score:', score)
