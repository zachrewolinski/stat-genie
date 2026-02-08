import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
_df = pd.read_csv('amtl.csv')

# Basic cleaning
_df = _df.copy()
_df['amtl_rate'] = _df['num_amtl'] / _df['sockets']
_df['human'] = (_df['genus'] == 'Homo sapiens').astype(int)

# Fit binomial GLM with binomial denominator via var_weights
formula = 'amtl_rate ~ human + age + prob_male + C(tooth_class)'
model = smf.glm(formula=formula, data=_df, family=sm.families.Binomial(), var_weights=_df['sockets'])
result = model.fit()

# Extract human coefficient
coef = result.params.get('human', np.nan)
se = result.bse.get('human', np.nan)
pval = result.pvalues.get('human', np.nan)

# Compute odds ratio and CI
or_val = float(np.exp(coef))
ci_low = float(np.exp(coef - 1.96 * se))
ci_high = float(np.exp(coef + 1.96 * se))

# Predicted rates at mean covariates for human vs nonhuman
mean_age = _df['age'].mean()
mean_prob_male = _df['prob_male'].mean()
# Use most common tooth_class for representative comparison
mode_tooth = _df['tooth_class'].mode().iloc[0]

pred_df = pd.DataFrame({
    'human': [0, 1],
    'age': [mean_age, mean_age],
    'prob_male': [mean_prob_male, mean_prob_male],
    'tooth_class': [mode_tooth, mode_tooth],
})
pred = result.predict(pred_df)

# Determine scalar conclusion
if np.isnan(coef):
    score = 0
else:
    effect = coef
    effect_norm = np.tanh(effect)
    if pval < 1e-6:
        sig = 1.0
    elif pval < 1e-4:
        sig = 0.9
    elif pval < 1e-3:
        sig = 0.8
    elif pval < 1e-2:
        sig = 0.6
    elif pval < 0.05:
        sig = 0.4
    else:
        sig = 0.2
    raw = effect_norm * sig
    score = int(round(raw * 100))

# Write scalar to conclusion.txt
with open('conclusion.txt', 'w') as f:
    f.write(str(score))

# Also print key results for inspection
print(result.summary())
print('Human coef:', coef)
print('OR:', or_val, 'CI:', (ci_low, ci_high), 'p:', pval)
print('Predicted rates (nonhuman, human):', pred.tolist())
print('Score:', score)
