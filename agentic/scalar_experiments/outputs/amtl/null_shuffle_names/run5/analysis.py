import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
_df = pd.read_csv('amtl.csv')

# Map columns based on metadata misalignment
# Inferred meaning from info.json and data patterns
# missing teeth count
_df = _df.rename(columns={
    'genus': 'num_missing',   # number of teeth missing of given class
    'age': 'num_sockets',     # number of observable sockets
    'pop': 'age_at_death',    # estimated age at death
    'num_amtl': 'stdev_age',  # assigned uncertainty of age at death
    'stdev_age': 'prob_male', # estimate of sex
    'tooth_class': 'genus',   # specimen genus
    'specimen': 'population', # region/population
    'prob_male': 'specimen_id',
    'sockets': 'tooth_class'
})

df = _df.copy()

# Basic validity checks
if (df['num_missing'] > df['num_sockets']).any():
    # This would be invalid for binomial counts
    # Cap at sockets to avoid negatives, but also flag
    df['num_missing'] = df[['num_missing', 'num_sockets']].min(axis=1)

# Create proportion
# Avoid zero sockets (shouldn't be zero, but guard)
df = df[df['num_sockets'] > 0].copy()
df['missing_prop'] = df['num_missing'] / df['num_sockets']

# Fit binomial GLM with logit link
formula = 'missing_prop ~ C(genus) + age_at_death + prob_male + C(tooth_class)'
model = smf.glm(
    formula=formula,
    data=df,
    family=sm.families.Binomial(),
    freq_weights=df['num_sockets']
).fit()

# Marginal standardization: predicted missing proportion by genus
species = df['genus'].unique()

preds = {}
for sp in species:
    tmp = df.copy()
    tmp['genus'] = sp
    preds[sp] = model.predict(tmp).mean()

# Compare Homo sapiens vs non-human primates
homo = 'Homo sapiens'
nonhuman = [sp for sp in species if sp != homo]

homo_pred = preds.get(homo, np.nan)
nonhuman_preds = [preds[sp] for sp in nonhuman]
nonhuman_mean = float(np.mean(nonhuman_preds))

diff = homo_pred - nonhuman_mean

# Bootstrap for uncertainty
rng = np.random.default_rng(42)
B = 500
boot_diffs = []

# Precompute indices for speed
n = len(df)
for _ in range(B):
    idx = rng.integers(0, n, size=n)
    sample = df.iloc[idx].copy()
    try:
        m = smf.glm(
            formula=formula,
            data=sample,
            family=sm.families.Binomial(),
            freq_weights=sample['num_sockets']
        ).fit()
    except Exception:
        continue
    sp_preds = {}
    for sp in species:
        tmp = sample.copy()
        tmp['genus'] = sp
        sp_preds[sp] = m.predict(tmp).mean()
    if homo in sp_preds:
        nh = [sp_preds[sp] for sp in nonhuman]
        boot_diffs.append(sp_preds[homo] - float(np.mean(nh)))

boot_diffs = np.array(boot_diffs)

# Compute statistics
prob_positive = float(np.mean(boot_diffs > 0)) if len(boot_diffs) else np.nan
ci_low, ci_high = (float(np.percentile(boot_diffs, 2.5)), float(np.percentile(boot_diffs, 97.5))) if len(boot_diffs) else (np.nan, np.nan)

# Save summary for decision
summary = {
    'homo_pred': homo_pred,
    'nonhuman_mean': nonhuman_mean,
    'diff': diff,
    'prob_positive': prob_positive,
    'ci_low': ci_low,
    'ci_high': ci_high,
    'boot_n': len(boot_diffs)
}

print('Model coef summary (partial):')
print(model.params)
print('\nAdjusted mean missing proportion by genus:')
print(preds)
print('\nSummary:')
print(summary)

# Map to Likert scale: scale by probability and effect size
# Use probability of positive difference, and standardized by magnitude
# If effect size small (<0.01), dampen.
if np.isnan(prob_positive):
    score = 0
else:
    # effect size factor
    mag = abs(diff)
    mag_factor = min(1.0, mag / 0.05)  # 5 percentage points as strong
    direction = 1 if diff > 0 else -1
    score = int(round(direction * prob_positive * 100 * mag_factor))

print('\nLikert score:', score)

# Write to conclusion.txt
with open('conclusion.txt', 'w') as f:
    f.write(str(score))
