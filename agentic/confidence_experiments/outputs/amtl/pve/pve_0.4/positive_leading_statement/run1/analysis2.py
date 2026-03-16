import pandas as pd
import numpy as np
import statsmodels.formula.api as smf


df = pd.read_csv('amtl.csv')

# Fit OLS with robust SE
model = smf.ols('num_amtl ~ C(genus) + age + prob_male + C(tooth_class)', data=df).fit(cov_type='HC3')

# Extract genus coefficients
coef_table = model.summary2().tables[1]

# Build marginal mean predictions for each genus using observed covariates
pred_means = {}
for genus in df['genus'].unique():
    df_tmp = df.copy()
    df_tmp['genus'] = genus
    pred = model.predict(df_tmp)
    pred_means[genus] = pred.mean()

# Compute differences vs Homo sapiens
ref = 'Homo sapiens'
ref_mean = pred_means[ref]

print('Adjusted mean predictions (marginal over observed covariates):')
for genus, mean in pred_means.items():
    print(f'  {genus}: {mean:.3f}')
print('Differences vs Homo sapiens:')
for genus, mean in pred_means.items():
    if genus == ref:
        continue
    print(f'  {genus} - {ref}: {mean - ref_mean:.3f}')

print('\nGenus coefficients (vs Homo sapiens):')
for genus in ['Pan','Papio','Pongo']:
    term = f'C(genus)[T.{genus}]'
    coef = coef_table.loc[term, 'Coef.']
    se = coef_table.loc[term, 'Std.Err.']
    p = coef_table.loc[term, 'P>|z|']
    ci_low = coef_table.loc[term, '[0.025']
    ci_high = coef_table.loc[term, '0.975]']
    print(f'  {term}: coef={coef:.3f}, SE={se:.3f}, p={p:.2e}, 95% CI=({ci_low:.3f},{ci_high:.3f})')

# Joint test of genus terms
# H0: all genus effects = 0
hypotheses = 'C(genus)[T.Pan] = 0, C(genus)[T.Papio] = 0, C(genus)[T.Pongo] = 0'
joint_test = model.f_test(hypotheses)
print('\nJoint F-test for genus effects:')
print(joint_test)
