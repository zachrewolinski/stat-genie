import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
df = pd.read_csv('affairs.csv')

# Basic summaries by children
summary = df.groupby('children')['affairs'].agg(['mean', 'median', 'std', 'count'])
print('Summary of affairs by children:')
print(summary)

# Proportion with any affairs (affairs > 0) by children
df['any_affair'] = (df['affairs'] > 0).astype(int)
prop_any = df.groupby('children')['any_affair'].mean()
print('\nProportion with any affairs by children:')
print(prop_any)

# Fit a Poisson regression for affair count with children as predictor, controlling for key covariates
# affairs is count-like but top-coded; Poisson is a simple starting point.

model_pois = smf.glm(
    formula='affairs ~ C(children) + age + yearsmarried + religiousness + education + occupation + rating',
    data=df,
    family=sm.families.Poisson()
).fit()

print('\nPoisson regression results (affairs count as outcome):')
print(model_pois.summary())

# Also fit a logistic regression for any affair
model_logit = smf.logit(
    formula='any_affair ~ C(children) + age + yearsmarried + religiousness + education + occupation + rating',
    data=df
).fit(disp=False)

print('\nLogistic regression results (any affair as outcome):')
print(model_logit.summary())

# Extract key effect of having children
# Children is coded yes/no; reference category depends on statsmodels ordering.
params_pois = model_pois.params
conf_pois = model_pois.conf_int()

params_logit = model_logit.params
conf_logit = model_logit.conf_int()

print('\nKey coefficients:')
for name, params, conf in [
    ('Poisson C(children)[T.yes]', params_pois, conf_pois),
    ('Logit C(children)[T.yes]', params_logit, conf_logit),
]:
    if 'C(children)[T.yes]' in params.index:
        est = params['C(children)[T.yes]']
        lo, hi = conf.loc['C(children)[T.yes]']
        print(f"{name}: est={est:.3f}, 95% CI=({lo:.3f}, {hi:.3f})")
    else:
        print(f"{name}: term not found; params index = {list(params.index)}")
