import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
raw = pd.read_csv('amtl.csv')

# Map shuffled columns to semantic names
# Based on observed values:
# - 'tooth_class' column contains genus labels
# - 'sockets' column contains tooth class labels
# - 'genus' column is integer counts (likely missing teeth)
# - 'age' column is integer counts (likely observable sockets)
# - 'pop' column is age-at-death (years)
# - 'stdev_age' column is probability male (0-1)

_df = raw.rename(
    columns={
        'tooth_class': 'genus_group',
        'sockets': 'tooth_class',
        'genus': 'num_missing',
        'age': 'n_sockets',
        'pop': 'age_years',
        'stdev_age': 'prob_male',
        'prob_male': 'specimen_id',
    }
)

# Drop invalid rows where missing exceeds sockets or sockets <= 0
_df = _df[_df['n_sockets'] > 0].copy()
_df = _df[_df['num_missing'] <= _df['n_sockets']].copy()

# Response as proportion with binomial weights
_df['prop_missing'] = _df['num_missing'] / _df['n_sockets']

# Fit binomial GLM
formula = 'prop_missing ~ C(genus_group) + age_years + prob_male + C(tooth_class)'
model = smf.glm(formula=formula, data=_df, family=sm.families.Binomial(), freq_weights=_df['n_sockets'])
result = model.fit()

# Prepare adjusted predictions
mean_age = _df['age_years'].mean()
mean_prob_male = _df['prob_male'].mean()

# Tooth class weights in sample
tooth_weights = _df['tooth_class'].value_counts(normalize=True).to_dict()

# Genus weights for non-human
nonhuman = ['Pan', 'Pongo', 'Papio']
nonhuman_weights = (
    _df.loc[_df['genus_group'].isin(nonhuman), 'genus_group']
    .value_counts(normalize=True)
    .to_dict()
)


def adjusted_mean_for_genus(genus_label: str) -> float:
    rows = []
    for tc, w in tooth_weights.items():
        rows.append({'genus_group': genus_label, 'age_years': mean_age, 'prob_male': mean_prob_male, 'tooth_class': tc, 'weight': w})
    pred_df = pd.DataFrame(rows)
    preds = result.predict(pred_df)
    return float((preds * pred_df['weight']).sum())


adj_means = {g: adjusted_mean_for_genus(g) for g in _df['genus_group'].unique()}

homo_mean = adj_means.get('Homo sapiens', np.nan)
nonhuman_mean = sum(adj_means[g] * nonhuman_weights.get(g, 0.0) for g in nonhuman)

diff = homo_mean - nonhuman_mean

# Bootstrap for uncertainty on diff
rng = np.random.default_rng(0)
boot_diffs = []

for _ in range(100):
    sample_idx = rng.integers(0, len(_df), size=len(_df))
    boot = _df.iloc[sample_idx].copy()
    try:
        boot_model = smf.glm(formula=formula, data=boot, family=sm.families.Binomial(), freq_weights=boot['n_sockets']).fit()
    except Exception:
        continue

    mean_age_b = boot['age_years'].mean()
    mean_prob_male_b = boot['prob_male'].mean()
    tooth_weights_b = boot['tooth_class'].value_counts(normalize=True).to_dict()
    nonhuman_weights_b = (
        boot.loc[boot['genus_group'].isin(nonhuman), 'genus_group']
        .value_counts(normalize=True)
        .to_dict()
    )

    def adj_mean_b(genus_label: str) -> float:
        rows = []
        for tc, w in tooth_weights_b.items():
            rows.append({'genus_group': genus_label, 'age_years': mean_age_b, 'prob_male': mean_prob_male_b, 'tooth_class': tc, 'weight': w})
        pred_df = pd.DataFrame(rows)
        preds = boot_model.predict(pred_df)
        return float((preds * pred_df['weight']).sum())

    try:
        adj_means_b = {g: adj_mean_b(g) for g in boot['genus_group'].unique()}
        homo_b = adj_means_b.get('Homo sapiens', np.nan)
        nonhuman_b = sum(adj_means_b[g] * nonhuman_weights_b.get(g, 0.0) for g in nonhuman)
        if np.isfinite(homo_b) and np.isfinite(nonhuman_b):
            boot_diffs.append(homo_b - nonhuman_b)
    except Exception:
        continue

boot_diffs = np.array(boot_diffs)
if boot_diffs.size:
    ci_low, ci_high = np.percentile(boot_diffs, [2.5, 97.5])
    prob_positive = float((boot_diffs > 0).mean())
else:
    ci_low = ci_high = np.nan
    prob_positive = 0.5

# Convert to Likert scalar
sign = 1 if diff > 0 else -1 if diff < 0 else 0
strength_prob = max(0.0, min(1.0, 2 * (prob_positive - 0.5)))
strength_diff = min(1.0, abs(diff) / 0.10)  # 10 percentage points as strong effect
score = int(round(100 * sign * strength_prob * strength_diff))

# Write conclusion
with open('conclusion.txt', 'w') as f:
    f.write(str(score))

# Print key results
print('Rows used:', len(_df))
print('Adjusted mean AMTL probability (Homo sapiens):', homo_mean)
print('Adjusted mean AMTL probability (non-human weighted):', nonhuman_mean)
print('Difference (Homo - nonhuman):', diff)
print('Bootstrap diffs:', boot_diffs.size)
print('Diff 95% CI:', (ci_low, ci_high))
print('Prob diff > 0:', prob_positive)
print('Conclusion score:', score)
