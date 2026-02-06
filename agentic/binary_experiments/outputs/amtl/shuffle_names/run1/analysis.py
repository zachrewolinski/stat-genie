import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from patsy import build_design_matrices

# Load data
raw = pd.read_csv('amtl.csv')

# Rename shuffled columns to their semantic meaning based on metadata
rename = {
    'sockets': 'tooth_class',   # Anterior/Posterior/Premolar
    'prob_male': 'specimen_id',
    'genus': 'num_missing',     # number of missing teeth in class
    'age': 'num_sockets',       # number of observable sockets in class
    'pop': 'age_est',           # estimated age at death
    'num_amtl': 'age_sd',       # uncertainty of age
    'stdev_age': 'prob_male',   # probability male (sex)
    'tooth_class': 'genus',     # genus category
    'specimen': 'population',
}

df = raw.rename(columns=rename).copy()

# Basic checks
if (df['num_missing'] > df['num_sockets']).any():
    raise ValueError('Found num_missing > num_sockets; check column mapping.')

# Response as AMTL rate with binomial weights
# Use Homo sapiens as the reference category
formula = 'amtl_rate ~ C(genus, Treatment(reference="Homo sapiens")) + age_est + prob_male + C(tooth_class)'

df['amtl_rate'] = df['num_missing'] / df['num_sockets']

model = smf.glm(
    formula=formula,
    data=df,
    family=sm.families.Binomial(),
    var_weights=df['num_sockets']
).fit()

# Predicted AMTL rates by genus at mean covariates, averaged across tooth classes
mean_age = df['age_est'].mean()
mean_prob_male = df['prob_male'].mean()

tooth_levels = sorted(df['tooth_class'].unique())
genus_levels = sorted(df['genus'].unique())

# Build prediction grid: for each genus, all tooth classes at mean age/sex
pred_rows = []
for g in genus_levels:
    for t in tooth_levels:
        pred_rows.append({'genus': g, 'tooth_class': t, 'age_est': mean_age, 'prob_male': mean_prob_male})

pred_df = pd.DataFrame(pred_rows)

# Get design matrix for prediction
pred_result = model.get_prediction(pred_df)

# Model parameters for bootstrap uncertainty
params = model.params.values
cov = model.cov_params().values

# Build design matrix for pred_df using model's design info
design_info = model.model.data.design_info
exog = build_design_matrices([design_info], pred_df)[0]
exog = np.asarray(exog)

# Function to compute mean predicted probability for each genus
# Average equally across tooth classes
n_tooth = len(tooth_levels)

def mean_pred_for_genus(beta):
    lin_pred = exog @ beta
    prob = 1 / (1 + np.exp(-lin_pred))
    probs_by_genus = {}
    idx = 0
    for g in genus_levels:
        probs = prob[idx:idx + n_tooth]
        probs_by_genus[g] = probs.mean()
        idx += n_tooth
    return probs_by_genus

# Point estimates
point_preds = mean_pred_for_genus(params)

# Parametric bootstrap for CIs
rng = np.random.default_rng(0)
num_draws = 10000
beta_draws = rng.multivariate_normal(params, cov, size=num_draws)

pred_samples = {g: np.empty(num_draws) for g in genus_levels}
for i in range(num_draws):
    preds = mean_pred_for_genus(beta_draws[i])
    for g in genus_levels:
        pred_samples[g][i] = preds[g]

# 95% CIs
pred_summary = {}
for g in genus_levels:
    mean = point_preds[g]
    lo, hi = np.percentile(pred_samples[g], [2.5, 97.5])
    pred_summary[g] = (mean, lo, hi)

# Pairwise differences: Homo sapiens - other genera
homo = 'Homo sapiens'

pairwise = {}
for g in genus_levels:
    if g == homo:
        continue
    diff_samples = pred_samples[homo] - pred_samples[g]
    diff_mean = point_preds[homo] - point_preds[g]
    lo, hi = np.percentile(diff_samples, [2.5, 97.5])
    pairwise[g] = (diff_mean, lo, hi)

# Decide answer: Yes if Homo has higher rate than all others with 95% CI above 0
higher_all = all(pairwise[g][1] > 0 for g in pairwise)

# Write conclusion
with open('conclusion.txt', 'w') as f:
    f.write('Yes\n' if higher_all else 'No\n')
    if higher_all:
        f.write('At mean age and sex, the model predicts higher AMTL rates for Homo sapiens than for Pan, Papio, and Pongo, and all pairwise 95% CIs for the differences are above zero. The binomial regression accounts for tooth class, age, and sex in these comparisons.\n')
    else:
        f.write('After accounting for age, sex, and tooth class, the model does not show Homo sapiens having a clearly higher AMTL rate than all non-human genera. At least one pairwise comparison has a 95% CI that includes zero.\n')

# Save a small summary for transparency
summary_lines = []
summary_lines.append('Predicted AMTL rates at mean age/sex (avg across tooth classes):')
for g in genus_levels:
    mean, lo, hi = pred_summary[g]
    summary_lines.append(f"  {g}: {mean:.3f} (95% CI {lo:.3f}, {hi:.3f})")
summary_lines.append('Pairwise differences (Homo sapiens - other genera):')
for g in pairwise:
    mean, lo, hi = pairwise[g]
    summary_lines.append(f"  vs {g}: {mean:.3f} (95% CI {lo:.3f}, {hi:.3f})")

with open('analysis_summary.txt', 'w') as f:
    f.write('\n'.join(summary_lines) + '\n')

# Also print to stdout for interactive checking
print('\n'.join(summary_lines))
