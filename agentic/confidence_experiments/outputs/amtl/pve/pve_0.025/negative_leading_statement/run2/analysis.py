import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

# Load data
path = 'amtl.csv'
df = pd.read_csv(path)

# Ensure categorical types and set reference level for genus
df['genus'] = pd.Categorical(df['genus'], categories=['Homo sapiens','Pan','Pongo','Papio'])
df['tooth_class'] = pd.Categorical(df['tooth_class'], categories=['Anterior','Premolar','Posterior'])

# Fit OLS with robust standard errors
model = smf.ols('num_amtl ~ C(genus) + age + prob_male + C(tooth_class)', data=df).fit(cov_type='HC3')

# Collect results for genus comparisons using model attributes
results = {}
for genus in ['Pan','Pongo','Papio']:
    term = f'C(genus)[T.{genus}]'
    if term in model.params.index:
        coef = model.params[term]
        se = model.bse[term]
        p = model.pvalues[term]
        ci_low = coef - 1.96 * se
        ci_high = coef + 1.96 * se
        results[genus] = {
            'coef': coef,
            'se': se,
            'p': p,
            'ci_low': ci_low,
            'ci_high': ci_high,
        }

# Save a small text report to help decide
print('N:', len(df))
print('Model R2:', model.rsquared)
print('Genus effects (negative means lower AMTL than Homo sapiens):')
for genus, res in results.items():
    print(genus, res)

# Also compute adjusted mean differences using predictive margins
# Predict for each row as Homo sapiens and as each genus, then average
base_df = df.copy()
means = {}
for genus in ['Homo sapiens','Pan','Pongo','Papio']:
    tmp = base_df.copy()
    tmp['genus'] = genus
    means[genus] = model.predict(tmp).mean()

print('Adjusted mean predictions by genus:')
for k,v in means.items():
    print(k, v)

# Differences Homo - others
for genus in ['Pan','Pongo','Papio']:
    diff = means['Homo sapiens'] - means[genus]
    print('Homo -', genus, diff)

# Additional model: Homo sapiens vs all non-human genera combined
df['is_human'] = (df['genus'] == 'Homo sapiens').astype(int)
model_human = smf.ols('num_amtl ~ is_human + age + prob_male + C(tooth_class)', data=df).fit(cov_type='HC3')
coef_human = model_human.params['is_human']
se_human = model_human.bse['is_human']
p_human = model_human.pvalues['is_human']
ci_low_human = coef_human - 1.96 * se_human
ci_high_human = coef_human + 1.96 * se_human

print('Homo vs non-human effect (is_human):', {
    'coef': coef_human,
    'se': se_human,
    'p': p_human,
    'ci_low': ci_low_human,
    'ci_high': ci_high_human,
})
