import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data

df = pd.read_csv('amtl.csv')

# Map columns to conceptual variables
# tooth_class (categorical): sockets column (Anterior/Posterior/Premolar)
# genus (categorical): tooth_class column (Homo sapiens, Pan, Papio, Pongo)
# age_at_death (continuous): pop column
# sex_prob (continuous 0-1): stdev_age column
# missing_teeth (count, noisy): genus column
# observable_sockets (count, noisy): age column

# Create analysis dataframe

analysis = pd.DataFrame({
    'tooth_class': df['sockets'],
    'genus': df['tooth_class'],
    'age_at_death': df['pop'],
    'sex_prob': df['stdev_age'],
    'missing': df['genus'],
    'sockets': df['age'],
})

# Clip missing to [0, sockets] to get feasible rates for binomial model
analysis['missing_clipped'] = analysis[['missing','sockets']].apply(lambda x: min(max(x['missing'], 0), x['sockets']), axis=1)
analysis['rate'] = analysis['missing_clipped'] / analysis['sockets']

# Basic summary of missing proportion by genus
summary = analysis.groupby('genus')['rate'].agg(['mean','std','count'])
print('Rate summary by genus')
print(summary)

# OLS on rate with weights = sockets
formula = 'rate ~ C(genus) + age_at_death + sex_prob + C(tooth_class)'
ols_model = smf.wls(formula, data=analysis, weights=analysis['sockets']).fit(cov_type='HC3')
print('\nOLS weighted (robust) summary (coef for genus):')
print(ols_model.summary().tables[1])

# GLM Binomial on clipped counts (using proportion and weights)
# use statsmodels GLM with Binomial and weights as sockets
glm_model = smf.glm(formula, data=analysis, family=sm.families.Binomial(), freq_weights=analysis['sockets']).fit()
print('\nGLM Binomial summary (coef for genus):')
print(glm_model.summary().tables[1])

# Extract effect for Homo sapiens vs baseline

# Determine baseline category
print('\nBaseline genus category:', analysis['genus'].astype('category').cat.categories[0])

# Compute estimated difference between Homo sapiens and baseline for both models

# If baseline is Homo sapiens, then we need to compare others; else coefficient for C(genus)[T.Homo sapiens]

for model, name in [(ols_model, 'OLS'), (glm_model, 'GLM')]:
    params = model.params
    if 'C(genus)[T.Homo sapiens]' in params:
        effect = params['C(genus)[T.Homo sapiens]']
        pval = model.pvalues['C(genus)[T.Homo sapiens]']
        print(f"{name} effect Homo sapiens vs baseline: coef={effect:.4f}, p={pval:.4g}")
    else:
        # baseline Homo sapiens; compare each other genus by negative of their coef
        print(f"{name} baseline is Homo sapiens. Other genus coefs:")
        for term in params.index:
            if term.startswith('C(genus)[T.'):
                print(term, params[term], model.pvalues[term])

