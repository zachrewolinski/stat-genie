import pandas as pd
import statsmodels.api as sm


df = pd.read_csv('crofoot.csv')

# Outcome
Y = df['m_focal']

# Group size (focal vs other)
rel_size = df['f_other'] - df['win']

# Location indicator: focal closer to own center than other is to its own
loc_focal_home = (df['m_other'] < df['n_focal']).astype(int)

X = pd.DataFrame({
    'rel_size': rel_size,
    'loc_focal_home': loc_focal_home
})
X = sm.add_constant(X)
model = sm.Logit(Y, X).fit(disp=False)
print(model.summary())

# Crosstab to show win rates by location
print('\nWin rates by loc_focal_home:')
print(pd.crosstab(Y, loc_focal_home, normalize='columns'))
