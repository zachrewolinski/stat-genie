import json
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

# Load data
_df = pd.read_csv('amtl.csv')

# Basic cleaning
_df = _df.copy()
_df['human'] = (_df['feature8'] == 'Homo sapiens').astype(int)

# Ensure numeric types
for col in ['feature3', 'feature5', 'feature7']:
    _df[col] = pd.to_numeric(_df[col], errors='coerce')

_df = _df.dropna(subset=['feature3', 'feature5', 'feature7', 'feature1', 'human'])

# OLS with robust SE; outcome is continuous (noisy counts)
model = smf.ols('feature3 ~ human + feature5 + feature7 + C(feature1)', data=_df).fit(cov_type='HC3')

coef = float(model.params['human'])
se = float(model.bse['human'])
pval = float(model.pvalues['human'])
ci_low, ci_high = model.conf_int().loc['human'].astype(float).tolist()

# Standardized effect size (approximate): divide coefficient by outcome SD
outcome_sd = float(_df['feature3'].std(ddof=0))
std_effect = coef / outcome_sd if outcome_sd > 0 else float('nan')

n = int(model.nobs)

# Map to Likert 0-100
# Heuristic: base on significance and effect size direction
if pval < 0.001:
    strength = 90
elif pval < 0.01:
    strength = 80
elif pval < 0.05:
    strength = 65
elif pval < 0.10:
    strength = 45
else:
    strength = 35

# Adjust for effect size magnitude (small/moderate/large)
abs_std = abs(std_effect)
if abs_std < 0.1:
    strength -= 10
elif abs_std < 0.3:
    strength += 0
elif abs_std < 0.5:
    strength += 5
else:
    strength += 10

# Directional mapping
if coef > 0:
    response = min(100, max(0, strength))
else:
    response = min(100, max(0, 100 - strength))

# Build explanation
explanation = (
    f"I fit a linear regression of AMTL (feature3) on a human indicator, age (feature5), "
    f"sex (feature7), and tooth class fixed effects (feature1). The human coefficient is "
    f"{coef:.3f} (SE {se:.3f}), 95% CI [{ci_low:.3f}, {ci_high:.3f}], p={pval:.4g} "
    f"with n={n}. The standardized effect is {std_effect:.3f} SD units. "
    f"A positive, statistically significant coefficient indicates higher AMTL in modern humans "
    f"after accounting for age, sex, and tooth class."
)

# Write conclusion
with open('conclusion.txt', 'w') as f:
    json.dump({"response": int(round(response)), "explanation": explanation}, f)
