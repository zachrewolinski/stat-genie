import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
path = 'amtl.csv'
df = pd.read_csv(path)

# Map columns based on inference
# missing counts
df['missing'] = df['genus']
# sockets count
df['sockets_n'] = df['age']
# age at death
df['age_years'] = df['pop']
# sex probability (prob male)
df['prob_male'] = df['stdev_age']
# tooth class category
# df['tooth_class'] already = sockets column
# genus category
# df['genus_cat'] already = tooth_class column

# Ensure valid binomial rows
valid = df['missing'].between(0, df['sockets_n'])
invalid_count = (~valid).sum()
print('Invalid rows (missing > sockets):', int(invalid_count))

# Drop invalid
df_valid = df[valid].copy()

# Build formula
# Use C() for categorical factors
df_valid['tooth_class_cat'] = df_valid['sockets']
df_valid['genus_cat'] = df_valid['tooth_class']

# Response as proportion with weights
# Avoid 0/0; sockets should be >0

# Fit GLM binomial with counts
endog = df_valid['missing'] / df_valid['sockets_n']
weights = df_valid['sockets_n']

formula = 'missing / sockets_n ~ C(genus_cat) + age_years + prob_male + C(tooth_class_cat)'
model = smf.glm(formula=formula, data=df_valid, family=sm.families.Binomial(), freq_weights=weights)
res = model.fit()
print(res.summary())

# Compute contrasts: Homo sapiens vs others (Pan, Pongo, Papio)
# Extract coefficients
params = res.params
cov = res.cov_params()

# Baseline genus is alphabetical order in C(); statsmodels uses first category as baseline
# We'll compute estimated differences for Homo sapiens vs each other genus.

# Get categories
cats = sorted(df_valid['genus_cat'].unique())
print('Genus categories:', cats)

# function to get coefficient name for a category
def cat_param(cat):
    return f'C(genus_cat)[T.{cat}]'

baseline = cats[0]
print('Baseline genus:', baseline)

# Compute log-odds difference of Homo sapiens vs others
hs = 'Homo sapiens'

# compute parameter for Homo sapiens relative to baseline
hs_coef = 0.0
if hs != baseline:
    hs_coef = params.get(cat_param(hs), 0.0)

# variance of hs coef
if hs != baseline:
    hs_var = cov.loc[cat_param(hs), cat_param(hs)]
else:
    hs_var = 0.0

results = []
for other in cats:
    if other == hs:
        continue
    # log-odds difference hs - other
    # if other is baseline: diff = hs_coef
    # else diff = hs_coef - coef_other
    if other == baseline:
        diff = hs_coef
        var = hs_var
    else:
        other_coef = params.get(cat_param(other), 0.0)
        diff = hs_coef - other_coef
        # variance
        var = hs_var + cov.loc[cat_param(other), cat_param(other)] - 2 * cov.loc[cat_param(hs), cat_param(other)]
    se = np.sqrt(var) if var >= 0 else np.nan
    z = diff / se if se and se > 0 else np.nan
    # two-sided p-value
    p = 2 * (1 - sm.stats.norm.cdf(abs(z))) if np.isfinite(z) else np.nan
    results.append((other, diff, se, z, p))

print('\nHomo sapiens vs others (log-odds difference):')
for other, diff, se, z, p in results:
    print(other, 'diff', diff, 'se', se, 'z', z, 'p', p)

# compute predicted marginal mean for each genus at mean covariates and reference tooth class
mean_age = df_valid['age_years'].mean()
mean_prob_male = df_valid['prob_male'].mean()

# choose reference tooth class (baseline)
tooth_cats = sorted(df_valid['tooth_class_cat'].unique())
base_tooth = tooth_cats[0]
print('Tooth classes:', tooth_cats, 'baseline', base_tooth)

# function to get linear predictor

def linpred(genus_cat, tooth_cat=base_tooth, age=mean_age, prob_male=mean_prob_male):
    lp = params['Intercept']
    # genus
    if genus_cat != baseline:
        lp += params.get(cat_param(genus_cat), 0.0)
    # tooth class
    if tooth_cat != base_tooth:
        lp += params.get(f'C(tooth_class_cat)[T.{tooth_cat}]', 0.0)
    lp += params.get('age_years', 0.0) * age
    lp += params.get('prob_male', 0.0) * prob_male
    return lp

print('\nPredicted AMTL probability at mean age/sex, base tooth class:')
for cat in cats:
    lp = linpred(cat)
    p = 1 / (1 + np.exp(-lp))
    print(cat, p)
