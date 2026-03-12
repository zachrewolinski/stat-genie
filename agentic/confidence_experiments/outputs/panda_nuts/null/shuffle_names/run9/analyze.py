import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
path = '/home/chenwang/stat-genie/agentic/scalar_experiments/outputs/panda_nuts/null/shuffle_names/run9/panda_nuts.csv'
df = pd.read_csv(path)

# Map shuffled columns to actual meaning based on metadata descriptions
# age column -> chimpanzee ID (unused in modeling)
# hammer column -> age in years
# nuts_opened column -> sex (f/m)
# sex column -> hammer type (unused in modeling)
# help column -> number of nuts opened
# chimpanzee column -> session duration in seconds
# seconds column -> received help (y/N)

df = df.rename(columns={
    'hammer': 'age_years',
    'nuts_opened': 'sex',
    'sex': 'hammer_type',
    'help': 'nuts_opened',
    'chimpanzee': 'seconds',
    'seconds': 'help'
})

# Clean / encode
# help: y/N -> 1/0
help_map = {'y': 1, 'N': 0}
df['help'] = df['help'].map(help_map)

# Remove any non-positive duration to avoid issues (none expected)
df = df[df['seconds'] > 0].copy()

# Poisson regression with log(seconds) offset for rate (nuts per second)
# predictors: age_years (continuous), sex (categorical), help (binary)

# Use statsmodels GLM
formula = 'nuts_opened ~ age_years + C(sex) + help'
model = smf.glm(
    formula=formula,
    data=df,
    family=sm.families.Poisson(),
    offset=np.log(df['seconds'])
).fit()

# Negative Binomial as robustness for overdispersion
nb_model = smf.glm(
    formula=formula,
    data=df,
    family=sm.families.NegativeBinomial(alpha=1.0),
    offset=np.log(df['seconds'])
).fit()

# Also compute simple efficiency and linear regression as robustness check
# Efficiency: nuts opened per second

df['efficiency'] = df['nuts_opened'] / df['seconds']
lin_model = smf.ols('efficiency ~ age_years + C(sex) + help', data=df).fit()

# Save key results to a JSON-like text file for later parsing
with open('/home/chenwang/stat-genie/agentic/scalar_experiments/outputs/panda_nuts/null/shuffle_names/run9/analysis_results.txt', 'w') as f:
    f.write('GLM Poisson summary:\n')
    f.write(str(model.summary()))
    f.write('\n\nGLM Negative Binomial summary:\n')
    f.write(str(nb_model.summary()))
    f.write('\n\nOLS efficiency summary:\n')
    f.write(str(lin_model.summary()))

print(model.summary())
print('\n')
print(nb_model.summary())
print('\n')
print(lin_model.summary())
