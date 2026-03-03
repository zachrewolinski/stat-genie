import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

# Load data

df = pd.read_csv('amtl.csv')

# Ensure categorical order with Homo sapiens as baseline
cat_order = ["Homo sapiens", "Pan", "Papio", "Pongo"]
df['genus'] = pd.Categorical(df['genus'], categories=cat_order)

# Fit OLS model with robust SE
formula = 'num_amtl ~ C(genus) + age + prob_male + C(tooth_class)'
model = smf.ols(formula, data=df).fit()
robust = model.get_robustcov_results(cov_type='HC3')

# Map parameter names to values
param_names = robust.model.exog_names
params = dict(zip(param_names, robust.params))
pvalues = dict(zip(param_names, robust.pvalues))

results = {}
for genus in ['Pan', 'Papio', 'Pongo']:
    key = f'C(genus)[T.{genus}]'
    results[genus] = {
        'coef': float(params.get(key, np.nan)),
        'pvalue': float(pvalues.get(key, np.nan))
    }

# Summaries for reporting
print('N', len(df))
print('Model formula', formula)
print('R2', robust.rsquared)
print('Genus coefficients vs Homo sapiens (robust HC3):')
for genus, res in results.items():
    print(genus, res)

# Compute standardized effect sizes (approx) for each contrast: coef / residual std
resid_std = np.sqrt(robust.mse_resid)
print('Residual SD', resid_std)
for genus, res in results.items():
    res['std_effect'] = res['coef'] / resid_std
    print('Std effect', genus, res['std_effect'])

# Check if all non-human coefficients negative and significant at 0.05
all_negative = all(res['coef'] < 0 for res in results.values())
all_sig = all(res['pvalue'] < 0.05 for res in results.values())
print('All non-human coefficients negative?', all_negative)
print('All non-human coefficients significant (p<0.05)?', all_sig)
