import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api as sm


df = pd.read_csv('amtl.csv')

# Keep relevant columns and drop missing values
cols = ['num_amtl', 'genus', 'age', 'prob_male', 'tooth_class']
sub = df[cols].dropna().copy()
sub['is_human'] = (sub['genus'] == 'Homo sapiens').astype(int)

# Model 1: human vs non-human
model = smf.ols('num_amtl ~ is_human + age + prob_male + C(tooth_class)', data=sub)
res = model.fit(cov_type='HC3')

# Model 2: genus categories (Homo as reference)
model2 = smf.ols('num_amtl ~ C(genus, Treatment(reference="Homo sapiens")) + age + prob_male + C(tooth_class)', data=sub)
res2 = model2.fit(cov_type='HC3')

# Adjusted mean difference using predictive margins: set genus to human vs non-human
# For non-human, keep original non-human genera; to create a counterfactual of non-human as a group,
# we set is_human to 0 while leaving other covariates the same.
sub_human = sub.copy()
sub_human['is_human'] = 1
sub_nonhuman = sub.copy()
sub_nonhuman['is_human'] = 0

pred_human = res.predict(sub_human).mean()
pred_nonhuman = res.predict(sub_nonhuman).mean()
mean_diff = pred_human - pred_nonhuman

# Summaries
print('N used:', len(sub))
print('Human vs Non-human OLS (HC3)')
print(res.summary().tables[1])
print('Adjusted mean num_amtl (human):', pred_human)
print('Adjusted mean num_amtl (non-human):', pred_nonhuman)
print('Adjusted mean difference (human - nonhuman):', mean_diff)

print('\nGenus-specific model (Homo reference) coefficients:')
print(res2.summary().tables[1])

# Extract key statistics
coef = res.params['is_human']
se = res.bse['is_human']
pval = res.pvalues['is_human']

print('\nKey stats:')
print('coef_is_human', coef)
print('se_is_human', se)
print('pval_is_human', pval)

