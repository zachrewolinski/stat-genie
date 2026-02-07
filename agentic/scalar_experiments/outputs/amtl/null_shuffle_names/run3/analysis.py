import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data

df = pd.read_csv('amtl.csv')

# Rename columns to clarify interpretation (names only for internal use)
df = df.rename(
    columns={
        'genus': 'num_missing',
        'age': 'num_sockets',
        'pop': 'age_est',
        'num_amtl': 'age_stdev',
        'stdev_age': 'prob_male',
        'tooth_class': 'genus_group',
        'sockets': 'tooth_class',
        'prob_male': 'specimen_id',
        'specimen': 'population',
    }
)

# Filter impossible rows (missing count cannot exceed sockets)
valid = df['num_missing'] <= df['num_sockets']
clean = df.loc[valid].copy()

# Build binomial regression: missing / sockets ~ genus + age + sex + tooth class
# Use proportions with frequency weights (num_sockets)
clean['missing_rate'] = clean['num_missing'] / clean['num_sockets']

model = smf.glm(
    formula='missing_rate ~ C(genus_group) + age_est + prob_male + C(tooth_class)',
    data=clean,
    family=sm.families.Binomial(),
    freq_weights=clean['num_sockets'],
).fit()

# Predict marginal mean missing rate for each genus at average covariates
mean_age = clean['age_est'].mean()
mean_prob_male = clean['prob_male'].mean()

pred_rows = []
for genus in clean['genus_group'].unique():
    for tclass in clean['tooth_class'].unique():
        pred_rows.append(
            {
                'genus_group': genus,
                'age_est': mean_age,
                'prob_male': mean_prob_male,
                'tooth_class': tclass,
            }
        )

pred_df = pd.DataFrame(pred_rows)

pred_df['pred'] = model.predict(pred_df)

# Average across tooth classes
mean_pred = pred_df.groupby('genus_group')['pred'].mean().sort_values(ascending=False)

# Compare Homo sapiens to non-human primates
homo_pred = mean_pred.loc['Homo sapiens']
non_human_pred = mean_pred.drop('Homo sapiens')

# Compute average difference (Homo - nonhuman mean)
nonhuman_mean = non_human_pred.mean()

# Get coefficient and p-value for Homo sapiens vs reference
# Use Papio as reference by releveling if present; otherwise use first.
if 'Papio' in clean['genus_group'].unique():
    clean_ref = clean.copy()
    clean_ref['genus_group'] = pd.Categorical(
        clean_ref['genus_group'],
        categories=['Papio'] + [g for g in clean_ref['genus_group'].unique() if g != 'Papio'],
        ordered=True,
    )
    model_ref = smf.glm(
        formula='missing_rate ~ C(genus_group) + age_est + prob_male + C(tooth_class)',
        data=clean_ref,
        family=sm.families.Binomial(),
        freq_weights=clean_ref['num_sockets'],
    ).fit()
    coef = model_ref.params.get('C(genus_group)[T.Homo sapiens]', np.nan)
    pval = model_ref.pvalues.get('C(genus_group)[T.Homo sapiens]', np.nan)
else:
    coef = model.params.get('C(genus_group)[T.Homo sapiens]', np.nan)
    pval = model.pvalues.get('C(genus_group)[T.Homo sapiens]', np.nan)

# Convert effect to scalar score
# Heuristic: combine sign, effect size, and significance
# Effect size: difference in predicted missing rate (percentage points)
pp_diff = (homo_pred - nonhuman_mean) * 100

# Scale: strong evidence if p < 0.01 and diff > 1pp
if np.isnan(pval):
    score = 0
else:
    if pp_diff > 0:
        if pval < 0.001:
            base = 85
        elif pval < 0.01:
            base = 70
        elif pval < 0.05:
            base = 55
        else:
            base = 20
    elif pp_diff < 0:
        if pval < 0.001:
            base = -85
        elif pval < 0.01:
            base = -70
        elif pval < 0.05:
            base = -55
        else:
            base = -20
    else:
        base = 0

    # Adjust by magnitude of pp_diff (cap at 30 points)
    mag_adj = min(30, abs(pp_diff) * 3)
    score = int(round(base + (mag_adj if base > 0 else -mag_adj)))

# Clamp to [-100, 100]
score = max(-100, min(100, score))

# Write conclusion
with open('conclusion.txt', 'w') as f:
    f.write(str(score))

# Also save a small report for inspection (not required)
report = {
    'n_total': len(df),
    'n_used': len(clean),
    'mean_pred': mean_pred.to_dict(),
    'homo_pred': float(homo_pred),
    'nonhuman_mean_pred': float(nonhuman_mean),
    'pp_diff': float(pp_diff),
    'coef_homo_vs_ref': float(coef) if not np.isnan(coef) else None,
    'pval_homo_vs_ref': float(pval) if not np.isnan(pval) else None,
    'score': int(score),
}

pd.Series(report).to_csv('analysis_report.csv')
