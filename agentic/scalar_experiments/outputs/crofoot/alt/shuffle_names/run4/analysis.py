import pandas as pd
import statsmodels.api as sm

pd.set_option('display.max_columns', None)

df = pd.read_csv('crofoot.csv')

# Define outcome and predictors
outcome = df['m_focal']

# Infer group sizes from metadata: f_other (focal group size) and win (other group size)
rel_size = df['f_other'] - df['win']

# Contest location: relative distance to each group's home range center
# m_other = distance of focal group from its center
# n_focal = distance of other group from its center
rel_location = df['n_focal'] - df['m_other']  # positive if contest closer to focal than other

X = pd.DataFrame({
    'rel_size': rel_size,
    'rel_location': rel_location,
})
X = sm.add_constant(X)

model = sm.GLM(outcome, X, family=sm.families.Binomial())
res = model.fit()
print(res.summary())

# Also check model with both distances separately
X2 = pd.DataFrame({
    'rel_size': rel_size,
    'dist_focal': df['m_other'],
    'dist_other': df['n_focal'],
})
X2 = sm.add_constant(X2)
res2 = sm.GLM(outcome, X2, family=sm.families.Binomial()).fit()
print('\nModel with separate distances:')
print(res2.summary())

# Simple correlations for context
print('\nCorrelation with outcome:')
for name, series in [('rel_size', rel_size), ('rel_location', rel_location), ('dist_focal', df['m_other']), ('dist_other', df['n_focal'])]:
    corr = series.corr(outcome)
    print(name, corr)
