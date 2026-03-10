import json
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
df = pd.read_csv('panda_nuts.csv')

# Basic cleaning
# Efficiency: nuts opened per second
df = df.copy()
df['efficiency'] = df['nuts_opened'] / df['seconds']

# Standardize categorical coding
df['sex'] = df['sex'].astype(str).str.lower()
df['help'] = df['help'].astype(str).str.lower()

# Remove any rows with missing values in key columns
df_model = df[['efficiency', 'nuts_opened', 'seconds', 'age', 'sex', 'help']].dropna()

# Linear model for efficiency
ols = smf.ols('efficiency ~ age + C(sex) + C(help)', data=df_model).fit(cov_type='HC3')

# Poisson model for counts with offset (rate model)
# Adds 1e-9 to seconds to avoid log(0) if any
df_model['log_seconds'] = np.log(df_model['seconds'].replace(0, 1e-9))
poisson = smf.glm(
    'nuts_opened ~ age + C(sex) + C(help)',
    data=df_model,
    family=sm.families.Poisson(),
    offset=df_model['log_seconds'],
).fit(cov_type='HC3')

# Collect results
def summarize_model(model):
    params = model.params
    pvals = model.pvalues
    return pd.DataFrame({'coef': params, 'p': pvals})

ols_summary = summarize_model(ols)
poisson_summary = summarize_model(poisson)

# Prepare text summaries for conclusion
summary = {
    'n': int(len(df_model)),
    'ols_r2': float(ols.rsquared),
    'ols': ols_summary.to_dict(),
    'poisson': poisson_summary.to_dict(),
}

# Save intermediate results for inspection
with open('analysis_results.json','w') as f:
    json.dump(summary, f, indent=2)

print(summary)
