import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
DF = pd.read_csv('amtl.csv')

# Basic cleaning
# Ensure numeric types
for col in ['num_amtl', 'sockets', 'age', 'stdev_age', 'prob_male']:
    DF[col] = pd.to_numeric(DF[col], errors='coerce')

# Drop rows with missing required fields
DF = DF.dropna(subset=['num_amtl', 'sockets', 'age', 'prob_male', 'tooth_class', 'genus'])

# Create human indicator
DF['is_human'] = (DF['genus'] == 'Homo sapiens').astype(int)

# Create failures for binomial
DF['num_present'] = DF['sockets'] - DF['num_amtl']

# Keep valid counts
DF = DF[(DF['num_amtl'] >= 0) & (DF['num_present'] >= 0)]

# Build GLM with binomial family using successes/failures
# Use tooth_class as categorical, adjust for age and sex (prob_male)

# Use formula with is_human and covariates
formula = 'num_amtl + num_present ~ is_human + age + prob_male + C(tooth_class)'

# statsmodels expects endog as a 2-column array for binomial counts
endog = DF[['num_amtl', 'num_present']]
exog = sm.add_constant(pd.get_dummies(DF[['is_human', 'age', 'prob_male', 'tooth_class']],
                                      columns=['tooth_class'], drop_first=True))

model = sm.GLM(endog, exog, family=sm.families.Binomial())
result = model.fit()

# Extract coefficient for is_human
coef = result.params['is_human']
se = result.bse['is_human']
pval = result.pvalues['is_human']

# Odds ratio and 95% CI
or_value = np.exp(coef)
ci_low, ci_high = np.exp(coef - 1.96*se), np.exp(coef + 1.96*se)

# Predicted difference at mean covariates
mean_vals = exog.mean()
mean_vals_human = mean_vals.copy()
mean_vals_human['is_human'] = 1.0
mean_vals_nonhuman = mean_vals.copy()
mean_vals_nonhuman['is_human'] = 0.0

pred_human = result.predict(mean_vals_human)
pred_nonhuman = result.predict(mean_vals_nonhuman)

print('n_rows', len(DF))
print('coef_is_human', coef)
print('se_is_human', se)
print('pval_is_human', pval)
print('odds_ratio', or_value)
print('or_ci_low', ci_low)
print('or_ci_high', ci_high)
print('pred_human', float(pred_human))
print('pred_nonhuman', float(pred_nonhuman))
print('pred_diff', float(pred_human - pred_nonhuman))
