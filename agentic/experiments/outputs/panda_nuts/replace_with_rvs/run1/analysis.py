import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
_df = pd.read_csv('panda_nuts.csv')

# Efficiency: nuts opened per second (descriptive)
_df['efficiency'] = _df['nuts_opened'] / _df['seconds']

# Poisson regression on counts with log(seconds) offset models rate directly
_df['log_seconds'] = np.log(_df['seconds'])
poisson_model = smf.glm(
    'nuts_opened ~ age + C(sex) + C(help)',
    data=_df,
    family=sm.families.Poisson(),
    offset=_df['log_seconds']
).fit(cov_type='HC0')

print(poisson_model.summary())
print('\nRobust (HC0) p-values:')
for name, pval in zip(poisson_model.model.exog_names, poisson_model.pvalues):
    print(f"{name}: {pval:.6f}")

# Save key results for interpretation
results = pd.DataFrame({
    'term': poisson_model.model.exog_names,
    'coef': poisson_model.params,
    'pval': poisson_model.pvalues
})
results.to_csv('analysis_results.csv', index=False)
