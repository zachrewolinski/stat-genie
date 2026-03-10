import pandas as pd
import statsmodels.api as sm
import numpy as np

# Load data
path = 'crofoot.csv'
df = pd.read_csv(path)

# Create variables
# Relative group size (focal - other)
df['rel_size'] = df['n_focal'] - df['n_other']
# Relative location: positive means contest is closer to focal home range center
# (other is farther from its center than focal is from its center)
df['rel_location'] = df['dist_other'] - df['dist_focal']

# Basic counts
summary = {
    'n': len(df),
    'win_rate': df['win'].mean(),
    'rel_size_mean': df['rel_size'].mean(),
    'rel_location_mean': df['rel_location'].mean(),
}

# Logistic regression
X = df[['rel_size', 'rel_location']]
X = sm.add_constant(X)
y = df['win']
model = sm.Logit(y, X)
res = model.fit(disp=False)

# Extract results
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

# Print
print('SUMMARY', summary)
print('\nLOGIT_RESULTS')
print(results)

# Also check if adding interaction improves? optional
# but not needed
