import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
info = json.load(open('info.json'))

df = pd.read_csv('amtl.csv')

# Keep relevant columns
cols = ['num_amtl', 'sockets', 'age', 'prob_male', 'tooth_class', 'genus']
missing_cols = [c for c in cols if c not in df.columns]
if missing_cols:
    raise ValueError(f"Missing columns: {missing_cols}")

df = df[cols].copy()

# Clean
for c in ['num_amtl', 'sockets', 'age', 'prob_male']:
    df[c] = pd.to_numeric(df[c], errors='coerce')

df = df.dropna(subset=cols)

df = df[df['sockets'] > 0].copy()

df['amtl_prop'] = df['num_amtl'] / df['sockets']

# Ensure categories
for c in ['tooth_class', 'genus']:
    df[c] = df[c].astype('category')

# Fit GLM (binomial with weights)
model = smf.glm(
    formula='amtl_prop ~ age + prob_male + C(tooth_class) + C(genus)',
    data=df,
    family=sm.families.Binomial(),
    freq_weights=df['sockets']
).fit()

# Prediction helper
weights = df['sockets'].to_numpy()

genera = list(df['genus'].cat.categories)
if 'Homo sapiens' not in genera:
    raise ValueError("Expected 'Homo sapiens' in genus categories")

pred_means = {}
for g in genera:
    tmp = df.copy()
    tmp['genus'] = g
    pred = model.predict(tmp)
    pred_means[g] = np.average(pred, weights=weights)

# Difference: Homo vs average of others
others = [g for g in genera if g != 'Homo sapiens']
mean_other = np.mean([pred_means[g] for g in others])
obs_diff = pred_means['Homo sapiens'] - mean_other

# Bootstrap for uncertainty
rng = np.random.default_rng(42)
B = 500
boot_diffs = []
for _ in range(B):
    idx = rng.integers(0, len(df), len(df))
    bdf = df.iloc[idx].copy()
    try:
        bmodel = smf.glm(
            formula='amtl_prop ~ age + prob_male + C(tooth_class) + C(genus)',
            data=bdf,
            family=sm.families.Binomial(),
            freq_weights=bdf['sockets']
        ).fit()
    except Exception:
        continue
    bweights = bdf['sockets'].to_numpy()
    bpred_means = {}
    for g in genera:
        tmp = bdf.copy()
        tmp['genus'] = g
        bpred = bmodel.predict(tmp)
        bpred_means[g] = np.average(bpred, weights=bweights)
    bmean_other = np.mean([bpred_means[g] for g in others])
    boot_diffs.append(bpred_means['Homo sapiens'] - bmean_other)

boot_diffs = np.array(boot_diffs)
if len(boot_diffs) < 50:
    raise RuntimeError("Too few bootstrap samples; check model stability")

boot_mean = float(np.mean(boot_diffs))
boot_sd = float(np.std(boot_diffs, ddof=1))
ci_low = float(np.quantile(boot_diffs, 0.025))
ci_high = float(np.quantile(boot_diffs, 0.975))

# Convert to Likert scalar [-100, 100]
# Use z-score style confidence mapping
z = boot_mean / boot_sd if boot_sd > 0 else 0.0
scalar = int(round(100 * np.tanh(z / 2)))

# If effect is opposite direction of question, keep sign
if obs_diff < 0 and scalar > 0:
    scalar = -scalar
if obs_diff > 0 and scalar < 0:
    scalar = -scalar

# Clamp to [-100, 100]
scalar = max(-100, min(100, scalar))

# Write conclusion
with open('conclusion.txt', 'w') as f:
    f.write(str(scalar))

# Print brief diagnostics for audit
print({
    'pred_means': pred_means,
    'obs_diff_homo_minus_other_mean': obs_diff,
    'boot_mean': boot_mean,
    'boot_sd': boot_sd,
    'ci_low': ci_low,
    'ci_high': ci_high,
    'scalar': scalar,
    'n': int(len(df)),
    'n_boot': int(len(boot_diffs))
})
