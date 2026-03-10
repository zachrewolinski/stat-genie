import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
path = 'panda_nuts.csv'
df = pd.read_csv(path)

# Basic cleaning
# Ensure seconds > 0 for rate calculations
_df = df[df['seconds'] > 0].copy()

# Poisson GLM with log link and offset for exposure time
_df['log_seconds'] = np.log(_df['seconds'])

# Use sex and help as categorical variables
model = smf.glm(
    formula='nuts_opened ~ age + C(sex) + C(help)',
    data=_df,
    family=sm.families.Poisson(),
    offset=_df['log_seconds']
).fit()

# Also compute rate for descriptive stats
_df['rate'] = _df['nuts_opened'] / _df['seconds']

summary = model.summary2().tables[1]

# Extract p-values for predictors
pvals = summary['P>|z|']
coefs = summary['Coef.']

# Save key outputs for interpretation
results = {
    'n': int(_df.shape[0]),
    'rate_mean': float(_df['rate'].mean()),
    'rate_sd': float(_df['rate'].std(ddof=1)),
    'coef_age': float(coefs.get('age', np.nan)),
    'p_age': float(pvals.get('age', np.nan)),
    'coef_sex_m': float(coefs.get('C(sex)[T.m]', np.nan)),
    'p_sex_m': float(pvals.get('C(sex)[T.m]', np.nan)),
    'coef_help_y': float(coefs.get('C(help)[T.y]', np.nan)),
    'p_help_y': float(pvals.get('C(help)[T.y]', np.nan)),
    'model_aic': float(model.aic),
}

print(results)
