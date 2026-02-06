import pandas as pd
import numpy as np
import statsmodels.api as sm

# Load data
Df = pd.read_csv('crofoot.csv')

# Map variables based on info.json descriptions (column names are shuffled)
# Outcome: 1 if focal won contest
Df['win_focal'] = Df['m_focal']

# Group sizes
Df['size_focal'] = Df['f_other']
Df['size_other'] = Df['win']

# Distances from home range centers
Df['dist_focal'] = Df['m_other']
Df['dist_other'] = Df['n_focal']

# Predictors
Df['rel_size'] = Df['size_focal'] - Df['size_other']
# Positive value means other group is farther from its own center (focal has home advantage)
Df['home_adv'] = Df['dist_other'] - Df['dist_focal']

# Fit logistic regression
X = Df[['rel_size', 'home_adv']]
X = sm.add_constant(X)
model = sm.Logit(Df['win_focal'], X)
result = model.fit(disp=False)

print(result.summary())

print("\nCoefficients:")
print(result.params)
print("\nOdds ratios:")
print(np.exp(result.params))
