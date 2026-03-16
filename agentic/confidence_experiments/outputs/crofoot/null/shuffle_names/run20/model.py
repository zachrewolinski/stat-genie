import pandas as pd
import numpy as np
import statsmodels.api as sm

# Load data
_df = pd.read_csv('crofoot.csv')

# Define variables based on metadata
# Outcome: m_focal (1 if focal won)
# Relative group size: f_other (size focal) - win (size other)
# Relative contest location: m_other (dist focal from center) - n_focal (dist other from center)
_df = _df.copy()
_df['rel_size'] = _df['f_other'] - _df['win']
_df['rel_dist'] = _df['m_other'] - _df['n_focal']

# logistic regression
X = sm.add_constant(_df[['rel_size','rel_dist']])
model = sm.Logit(_df['m_focal'], X).fit(disp=False)
print(model.summary())

# Odds ratios and p-values
params = model.params
pvals = model.pvalues
odds = np.exp(params)
print('\nOdds ratios:')
print(odds)
print('\nP-values:')
print(pvals)

# Also model with separate sizes/distances
X2 = sm.add_constant(_df[['f_other','win','m_other','n_focal']])
model2 = sm.Logit(_df['m_focal'], X2).fit(disp=False)
print('\nModel with separate sizes and distances:')
print(model2.summary())
print('\nP-values model2:')
print(model2.pvalues)

# simple logistic with rel_size only, rel_dist only
for var in ['rel_size','rel_dist']:
    Xv = sm.add_constant(_df[[var]])
    mv = sm.Logit(_df['m_focal'], Xv).fit(disp=False)
    print(f"\nModel with {var} only:")
    print(mv.summary())
    print('P-values', mv.pvalues)

