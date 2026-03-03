import pandas as pd
import statsmodels.formula.api as smf
import numpy as np

# Load data
amtl = pd.read_csv('amtl.csv')

# Fit OLS with categorical genus, tooth_class; adjust for age and prob_male
# Use Homo sapiens as reference automatically by pandas if alphabetical? We'll set explicitly.
# Ensure category order
amtl['genus'] = pd.Categorical(amtl['genus'], categories=['Homo sapiens','Pan','Pongo','Papio'])
# If categories missing? We'll drop NA.

formula = 'num_amtl ~ C(genus) + age + prob_male + C(tooth_class)'
model = smf.ols(formula, data=amtl).fit(cov_type='cluster', cov_kwds={'groups': amtl['specimen']})
print(model.summary())

# Extract coefficients for non-human genera
coef = model.params
se = model.bse
p = model.pvalues

for g in ['C(genus)[T.Pan]','C(genus)[T.Pongo]','C(genus)[T.Papio]']:
    if g in coef:
        print(g, coef[g], se[g], p[g])

# Compute average non-human difference vs Homo sapiens
# Average of three genus coefficients (Pan, Pongo, Papio)
coefs = np.array([coef.get('C(genus)[T.Pan]',0), coef.get('C(genus)[T.Pongo]',0), coef.get('C(genus)[T.Papio]',0)])

# Delta method for mean coefficient
d = np.array([1/3,1/3,1/3])
# build covariance of relevant coefficients
cov = model.cov_params()
idx = ['C(genus)[T.Pan]','C(genus)[T.Pongo]','C(genus)[T.Papio]']
# Ensure order
cov_sub = cov.loc[idx, idx]
mean_coef = d @ coefs
mean_se = np.sqrt(d @ cov_sub.values @ d)

# t-stat
from scipy import stats

t_stat = mean_coef / mean_se
p_val = 2 * (1 - stats.norm.cdf(abs(t_stat)))

print('Average non-human coef (relative to Homo sapiens):', mean_coef)
print('SE', mean_se, 't', t_stat, 'p', p_val)

# compute adjusted means for each genus at mean age and prob_male and reference tooth_class (anterior?)
# We'll compute marginal means by averaging over observed distribution of covariates and tooth_class.

# Use model to predict for each observation, setting genus to each level
levels = ['Homo sapiens','Pan','Pongo','Papio']

pred_means = {}
for g in levels:
    df = amtl.copy()
    df['genus'] = g
    pred = model.predict(df)
    pred_means[g] = pred.mean()

print('Predicted mean num_amtl by genus (marginal over covariates):')
for g, m in pred_means.items():
    print(g, m)

# Compare Homo sapiens to mean of non-human predicted means
nonhuman_mean = np.mean([pred_means[g] for g in ['Pan','Pongo','Papio']])
print('Homo sapiens mean', pred_means['Homo sapiens'], 'Non-human mean', nonhuman_mean, 'Difference', pred_means['Homo sapiens']-nonhuman_mean)

