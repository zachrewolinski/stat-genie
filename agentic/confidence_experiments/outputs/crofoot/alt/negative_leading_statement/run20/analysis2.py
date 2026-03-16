import pandas as pd
import statsmodels.api as sm
import numpy as np

# Load data
path = 'crofoot.csv'
df = pd.read_csv(path)

# Logistic regression with distances separately
X = df[['dist_focal', 'dist_other', 'n_focal', 'n_other']]
X = sm.add_constant(X)
y = df['win']
model = sm.Logit(y, X)
res = model.fit(disp=False)

params = res.params
conf = res.conf_int()
conf.columns = ['ci_low', 'ci_high']
se = res.bse
pvals = res.pvalues

odds_ratios = np.exp(params)
ci_or = np.exp(conf)

results = pd.DataFrame({
    'coef': params,
    'se': se,
    'pval': pvals,
    'odds_ratio': odds_ratios,
    'or_ci_low': ci_or['ci_low'],
    'or_ci_high': ci_or['ci_high'],
})

print('LOGIT_RESULTS')
print(results)
