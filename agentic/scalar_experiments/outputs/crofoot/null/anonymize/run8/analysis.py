import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tools import add_constant

# Load data

df = pd.read_csv('crofoot.csv')

# Variables
# Outcome: 1 if focal won
outcome = df['feature4']

# Relative group size (focal - other)
rel_group_size = df['feature7'] - df['feature8']

# Location advantage: positive when focal is closer to its home range center
location_adv = df['feature6'] - df['feature5']

# Standardize predictors for comparable effect sizes
rel_group_size_z = (rel_group_size - rel_group_size.mean()) / rel_group_size.std(ddof=0)
location_adv_z = (location_adv - location_adv.mean()) / location_adv.std(ddof=0)

X = pd.DataFrame({
    'rel_group_size_z': rel_group_size_z,
    'location_adv_z': location_adv_z,
})
X = add_constant(X)

model = sm.GLM(outcome, X, family=sm.families.Binomial())
res = model.fit()

print(res.summary())

# Odds ratios and CI
params = res.params
conf = res.conf_int()
odds = np.exp(params)
conf_odds = np.exp(conf)

print('\nOdds ratios (per 1 SD):')
for name in ['rel_group_size_z', 'location_adv_z']:
    print(f"{name}: OR={odds[name]:.3f}, 95% CI=({conf_odds.loc[name,0]:.3f}, {conf_odds.loc[name,1]:.3f}), p={res.pvalues[name]:.4f}")

# Also test raw predictors for interpretability (per individual and per meter)
X_raw = pd.DataFrame({
    'rel_group_size': rel_group_size,
    'location_adv': location_adv,
})
X_raw = add_constant(X_raw)
model_raw = sm.GLM(outcome, X_raw, family=sm.families.Binomial())
res_raw = model_raw.fit()

print('\nRaw predictor model:')
print(res_raw.summary())

# Compute predicted probabilities at +/-1 SD for each predictor holding the other at 0
mean_rel = 0.0
mean_loc = 0.0
sd_rel = rel_group_size.std(ddof=0)
sd_loc = location_adv.std(ddof=0)

# Using raw model for interpretability
b0, b_rel, b_loc = res_raw.params

def logistic(x):
    return 1 / (1 + np.exp(-x))

# Baseline at 0 (equal size, equal location distance)
base = logistic(b0 + b_rel * 0 + b_loc * 0)

# +/- 1 SD changes
p_rel_plus = logistic(b0 + b_rel * sd_rel + b_loc * 0)
p_rel_minus = logistic(b0 + b_rel * (-sd_rel) + b_loc * 0)

p_loc_plus = logistic(b0 + b_rel * 0 + b_loc * sd_loc)
p_loc_minus = logistic(b0 + b_rel * 0 + b_loc * (-sd_loc))

print('\nPredicted win probability (holding other predictor at 0):')
print(f"Baseline (equal size/location): {base:.3f}")
print(f"Rel size +1 SD: {p_rel_plus:.3f}, -1 SD: {p_rel_minus:.3f}")
print(f"Location adv +1 SD: {p_loc_plus:.3f}, -1 SD: {p_loc_minus:.3f}")

# Sample size
print(f"\nN={len(df)}")
