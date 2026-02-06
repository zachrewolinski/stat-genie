import pandas as pd
import numpy as np
import statsmodels.api as sm

# Load data
_df = pd.read_csv('crofoot.csv')

# Relative group size: focal minus other
_df['rel_size'] = _df['n_focal'] - _df['n_other']

# Contest location: difference in distance to each group's home-range center.
# Positive values mean the contest is closer to the focal group's center.
_df['rel_location'] = _df['dist_other'] - _df['dist_focal']

# Fit logistic regression: win ~ relative size + relative location
X = sm.add_constant(_df[['rel_size', 'rel_location']])
model = sm.Logit(_df['win'], X).fit(disp=False)

print(model.summary())

# Odds ratios for interpretability
params = model.params
or_rel_size = float(np.exp(params['rel_size']))
or_rel_location_per_100m = float(np.exp(params['rel_location'] * 100))

print('\nOdds ratios:')
print(f"  rel_size (per +1 individual): {or_rel_size:.3f}")
print(f"  rel_location (per +100 m closer to focal): {or_rel_location_per_100m:.3f}")

# Save key results for downstream review if needed
results = pd.DataFrame({
    'coef': model.params,
    'std_err': model.bse,
    'z': model.tvalues,
    'p_value': model.pvalues,
})
results.to_csv('model_results.csv', index=True)
