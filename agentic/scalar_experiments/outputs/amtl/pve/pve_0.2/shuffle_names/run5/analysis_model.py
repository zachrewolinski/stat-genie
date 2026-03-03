import pandas as pd
import statsmodels.formula.api as smf
import numpy as np

# load
_df = pd.read_csv('amtl.csv')

# rename columns for clarity (mapping guesses)
# outcome: genus -> amtl_rate (logit or transformed)
# genus category: tooth_class
# tooth class: sockets
# age at death: pop
# sex: stdev_age
# specimen id: prob_male

df = _df.copy()

df = df.rename(columns={
    'genus': 'amtl_metric',
    'tooth_class': 'genus_cat',
    'sockets': 'tooth_class_cat',
    'pop': 'age_est',
    'stdev_age': 'prob_male',
    'prob_male': 'specimen_id',
    'age': 'sockets_n',
    'num_amtl': 'age_uncert'
})

# model
formula = 'amtl_metric ~ C(genus_cat) + age_est + prob_male + C(tooth_class_cat)'
model = smf.ols(formula, data=df).fit(cov_type='cluster', cov_kwds={'groups': df['specimen_id']})

print(model.summary())

# get predicted mean by genus at average covariates
mean_age = df['age_est'].mean()
mean_prob_male = df['prob_male'].mean()
# use most common tooth_class_cat (tooth class)
mode_tooth_class = df['tooth_class_cat'].mode()[0]

preds = []
for g in df['genus_cat'].unique():
    row = {
        'genus_cat': g,
        'age_est': mean_age,
        'prob_male': mean_prob_male,
        'tooth_class_cat': mode_tooth_class
    }
    pred = model.predict(pd.DataFrame([row]))[0]
    preds.append((g, pred))

print('\nPredicted mean amtl_metric by genus (at mean covariates, tooth_class=%s):' % mode_tooth_class)
for g, p in preds:
    print(g, p)

# pairwise differences Homo vs each non-human
# Use model to compute differences and standard errors via linear hypotheses
import statsmodels.api as sm

# base category is alphabetical by default; get params
print('\nParams:', model.params)

# Use t_test for contrasts
params = model.params.index

# Build contrast for Homo vs each other genus
homo = 'Homo sapiens'

genera = [g for g in df['genus_cat'].unique() if g != homo]

for g in genera:
    # We need to account for reference category
    # If homo is reference, contrast is negative of other coef
    # If other is reference, contrast is coef of homo
    # Otherwise, difference is coef_homo - coef_g
    coef_names = list(params)
    L = np.zeros(len(coef_names))
    # intercept and other covariates cancel out
    # for categorical: statsmodels uses Treatment coding with baseline as first sorted category
    # names like C(genus_cat)[T.Pan]
    if f'C(genus_cat)[T.{homo}]' in coef_names:
        L[coef_names.index(f'C(genus_cat)[T.{homo}]')] = 1
    if f'C(genus_cat)[T.{g}]' in coef_names:
        L[coef_names.index(f'C(genus_cat)[T.{g}]')] = -1
    # If homo is baseline, its coef not in params; then L has -1 for g only
    # If g is baseline, its coef not in params; L has +1 for homo only
    ttest = model.t_test(L)
    diff = float(ttest.effect)
    pval = float(ttest.pvalue)
    print(f'Contrast Homo - {g}: diff={diff:.4f}, p={pval:.4g}')

