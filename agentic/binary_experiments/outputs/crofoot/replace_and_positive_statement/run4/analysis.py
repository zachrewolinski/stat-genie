import pandas as pd
import statsmodels.formula.api as smf

# Load data
Df = pd.read_csv('crofoot.csv')

# Create predictors
Df['rel_size'] = Df['n_focal'] - Df['n_other']
Df['rel_dist'] = Df['dist_other'] - Df['dist_focal']  # positive means contest is closer to focal than other

# Fit logistic regression: win ~ relative size + relative location
model = smf.logit('win ~ rel_size + rel_dist', data=Df).fit(disp=False)

# Print summary to stdout for inspection
print(model.summary())

# Extract key stats for programmatic use if needed
params = model.params
pvalues = model.pvalues
print('\nKey coefficients:')
for name in ['rel_size', 'rel_dist']:
    print(f"{name}: coef={params[name]:.4f}, p={pvalues[name]:.4f}")
