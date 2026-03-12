import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
path = 'panda_nuts.csv'
df = pd.read_csv(path)

# Basic derived columns
# Efficiency as rate: nuts opened per second
# (Used only for descriptive stats; modeling uses counts with exposure)
df['rate'] = df['nuts_opened'] / df['seconds']
df['log_seconds'] = np.log(df['seconds'])

# Poisson GLM with exposure (offset)
formula = 'nuts_opened ~ age + C(sex) + C(help)'
poisson_model = smf.glm(
    formula=formula,
    data=df,
    family=sm.families.Poisson(),
    offset=df['log_seconds']
).fit()

# Overdispersion check
overdispersion = poisson_model.deviance / poisson_model.df_resid

# Negative Binomial (discrete) with exposure to handle overdispersion
# Use the discrete model which estimates alpha
try:
    nb_model = smf.negativebinomial(
        formula=formula,
        data=df,
        exposure=df['seconds']
    ).fit(disp=False)
    nb_available = True
except Exception as e:
    nb_model = None
    nb_available = False
    nb_error = str(e)

# Extract results
results = {
    'n': len(df),
    'rate_mean': df['rate'].mean(),
    'rate_std': df['rate'].std(),
    'poisson_overdispersion': overdispersion,
    'poisson_params': poisson_model.params.to_dict(),
    'poisson_pvalues': poisson_model.pvalues.to_dict(),
    'poisson_irr': np.exp(poisson_model.params).to_dict(),
}

if nb_available:
    results['nb_params'] = nb_model.params.to_dict()
    results['nb_pvalues'] = nb_model.pvalues.to_dict()
    results['nb_irr'] = np.exp(nb_model.params).to_dict()
    # alpha parameter for NB
    if 'alpha' in nb_model.params.index:
        results['nb_alpha'] = nb_model.params['alpha']

print('N:', results['n'])
print('Mean rate (nuts/sec):', results['rate_mean'])
print('Std rate:', results['rate_std'])
print('Poisson overdispersion (deviance/df_resid):', results['poisson_overdispersion'])
print('\nPoisson coefficients (IRR, p-values):')
for term in poisson_model.params.index:
    print(f"  {term}: IRR={results['poisson_irr'][term]:.3f}, p={results['poisson_pvalues'][term]:.4f}")

if nb_available:
    print('\nNegative Binomial coefficients (IRR, p-values):')
    for term in nb_model.params.index:
        if term == 'alpha':
            continue
        print(f"  {term}: IRR={results['nb_irr'][term]:.3f}, p={results['nb_pvalues'][term]:.4f}")
    if 'nb_alpha' in results:
        print('NB alpha:', results['nb_alpha'])
else:
    print('NB model not available:', nb_error)

# Save results for later inspection if needed
pd.Series(results).to_json('analysis_results.json', orient='index')
