import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data

df = pd.read_csv('mortgage.csv')

# key variables
# feature2: female indicator (1 female,0 male)
# feature14: accepted (1 accepted,0 denied)
# feature11: denied

# Basic counts
n_total = len(df)

# Approval rate by gender
rates = df.groupby('feature2')['feature14'].mean()
counts = df.groupby('feature2')['feature14'].agg(['count','sum'])

# Two-proportion z-test for approval rate difference
# Use statsmodels proportion ztest
from statsmodels.stats.proportion import proportions_ztest

successes = counts['sum'].values
nobs = counts['count'].values
zstat, pval = proportions_ztest(successes, nobs)

# Logistic regression: approval ~ female (unadjusted)
model_unadj = smf.logit('feature14 ~ feature2', data=df).fit(disp=False)

# Adjusted model with relevant covariates
# include creditworthiness and other controls
# We'll use features 1-13 excluding 11 and 14 and 2 maybe? but include feature2 + others
covariates = [
    'feature1','feature3','feature4','feature5','feature6','feature7','feature8',
    'feature9','feature10','feature12','feature13'
]
formula = 'feature14 ~ feature2 + ' + ' + '.join(covariates)
model_adj = smf.logit(formula, data=df).fit(disp=False)

# Compute marginal effect? We'll compute odds ratio for feature2
params_unadj = model_unadj.params
conf_unadj = model_unadj.conf_int()

params_adj = model_adj.params
conf_adj = model_adj.conf_int()

# Convert to odds ratios
or_unadj = np.exp(params_unadj['feature2'])
ci_unadj = np.exp(conf_unadj.loc['feature2'])

or_adj = np.exp(params_adj['feature2'])
ci_adj = np.exp(conf_adj.loc['feature2'])

# Print summary
print('n_total', n_total)
print('approval rates by gender (0=male,1=female):')
print(rates)
print('counts (count, sum accepted):')
print(counts)
print('two-proportion z-test: z=%.4f, p=%.6g' % (zstat, pval))
print('unadjusted logit coef feature2: %.4f, p=%.6g' % (params_unadj['feature2'], model_unadj.pvalues['feature2']))
print('unadjusted OR: %.4f, 95%% CI [%.4f, %.4f]' % (or_unadj, ci_unadj[0], ci_unadj[1]))
print('adjusted logit coef feature2: %.4f, p=%.6g' % (params_adj['feature2'], model_adj.pvalues['feature2']))
print('adjusted OR: %.4f, 95%% CI [%.4f, %.4f]' % (or_adj, ci_adj[0], ci_adj[1]))

# Also compute predicted approval rates at means? We'll compute average marginal effect for gender using get_margeff
try:
    margeff = model_adj.get_margeff(at='overall', method='dydx')
    print(margeff.summary())
except Exception as e:
    print('margeff error', e)

