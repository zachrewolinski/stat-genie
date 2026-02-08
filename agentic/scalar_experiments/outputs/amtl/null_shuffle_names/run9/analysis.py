import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
raw = pd.read_csv('amtl.csv')

# Map columns based on observed values
# sockets -> tooth class (Anterior/Posterior/Premolar)
# tooth_class -> genus (Homo sapiens, Pan, Papio, Pongo)
# genus -> missing teeth count (integer 0-12)
# age -> total observable sockets (integer 2-14)
# pop -> estimated age at death (years)
# stdev_age -> prob_male (0-1)
# prob_male -> specimen id (not used in model)

df = raw.copy()

rename_map = {
    'sockets': 'tooth_class',
    'tooth_class': 'genus_name',
    'genus': 'missing_teeth',
    'age': 'total_sockets',
    'pop': 'age_years',
    'stdev_age': 'prob_male',
    'prob_male': 'specimen_id',
}

df = df.rename(columns=rename_map)

# Basic cleaning
valid = (
    (df['total_sockets'] > 0)
    & (df['missing_teeth'] >= 0)
    & (df['missing_teeth'] <= df['total_sockets'])
)

df = df[valid].copy()

# Create indicator for human

df['is_human'] = (df['genus_name'] == 'Homo sapiens').astype(int)

# Ensure category type for tooth class

df['tooth_class'] = df['tooth_class'].astype('category')

# Response as proportion with binomial weights

df['missing_rate'] = df['missing_teeth'] / df['total_sockets']

# Fit binomial GLM
formula = 'missing_rate ~ is_human + age_years + prob_male + C(tooth_class)'
model = smf.glm(
    formula=formula,
    data=df,
    family=sm.families.Binomial(),
    freq_weights=df['total_sockets'],
).fit()

# Marginal standardized difference: set is_human to 1 vs 0 for all rows

def predict_with_is_human(flag: int) -> np.ndarray:
    tmp = df.copy()
    tmp['is_human'] = flag
    return model.predict(tmp)

pred_human = predict_with_is_human(1)
pred_nonhuman = predict_with_is_human(0)

mean_diff = float(np.mean(pred_human - pred_nonhuman))

# Bootstrap CI for mean difference
rng = np.random.default_rng(42)
B = 500
boot_diffs = []

n = len(df)
for _ in range(B):
    idx = rng.integers(0, n, n)
    sample = df.iloc[idx]
    try:
        boot_model = smf.glm(
            formula=formula,
            data=sample,
            family=sm.families.Binomial(),
            freq_weights=sample['total_sockets'],
        ).fit()
        tmp_h = sample.copy()
        tmp_n = sample.copy()
        tmp_h['is_human'] = 1
        tmp_n['is_human'] = 0
        diff = float(np.mean(boot_model.predict(tmp_h) - boot_model.predict(tmp_n)))
        boot_diffs.append(diff)
    except Exception:
        continue

boot_diffs = np.array(boot_diffs)

if len(boot_diffs) > 10:
    se = float(np.std(boot_diffs, ddof=1))
    ci_low, ci_high = np.quantile(boot_diffs, [0.025, 0.975])
else:
    se = float('nan')
    ci_low, ci_high = float('nan'), float('nan')

# Convert to Likert-style scalar -100..100
# Use z-score from bootstrap; map via tanh to cap at 100
if np.isfinite(se) and se > 0:
    z = mean_diff / se
else:
    z = 0.0

score = int(np.round(100 * np.tanh(z / 2)))
score = max(-100, min(100, score))

# Output diagnostics
print('Rows used:', len(df))
print('Mean diff (human - nonhuman):', mean_diff)
print('Bootstrap SE:', se)
print('95% CI:', (ci_low, ci_high))
print('Z:', z)
print('Score:', score)

# Write conclusion
with open('conclusion.txt', 'w', encoding='utf-8') as f:
    f.write(str(score))
