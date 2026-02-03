import numpy as np
import pandas as pd
import statsmodels.api as sm

# Load data
df = pd.read_csv('crofoot.csv')

# Define predictors
# Relative group size: focal size minus other size
rel_size = df['n_focal'] - df['n_other']
# Location advantage: positive if contest closer to focal home range center
loc_adv = df['dist_other'] - df['dist_focal']

X = pd.DataFrame({
    'rel_size': rel_size,
    'loc_adv': loc_adv,
})
X = sm.add_constant(X)

# Outcome
y = df['win']

# Logistic regression
model = sm.Logit(y, X)
result = model.fit(disp=False)

print(result.summary())

# Also compute odds ratios for interpretability
odds_ratios = result.params.apply(lambda b: float(np.exp(b)))
print('\nOdds ratios:')
print(odds_ratios)
