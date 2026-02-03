import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
_df = pd.read_csv('panda_nuts.csv')

# Map shuffled columns to their described meanings
# According to info.json descriptions:
# age -> chimpanzee_id, hammer -> age_years, nuts_opened -> sex, sex -> hammer_type,
# help -> nuts_opened_count, chimpanzee -> session_seconds, seconds -> received_help
_df = _df.rename(
    columns={
        'age': 'chimpanzee_id',
        'hammer': 'age_years',
        'nuts_opened': 'sex',
        'sex': 'hammer_type',
        'help': 'nuts_opened_count',
        'chimpanzee': 'session_seconds',
        'seconds': 'received_help'
    }
)

# Basic cleaning
_df['received_help'] = _df['received_help'].astype(str).str.strip().str.lower().map({'y': 1, 'yes': 1, 'n': 0, 'no': 0})
_df['sex'] = _df['sex'].astype(str).str.strip().str.lower().map({'m': 'm', 'f': 'f'})

# Drop any rows with missing key fields
_df = _df.dropna(subset=['nuts_opened_count', 'session_seconds', 'age_years', 'sex', 'received_help'])

# Efficiency = nuts opened per second
_df['efficiency'] = _df['nuts_opened_count'] / _df['session_seconds']

# Model nuts opened as a rate with exposure = session_seconds
# Poisson GLM with log link and offset log(seconds) to model efficiency
_df['log_seconds'] = np.log(_df['session_seconds'])
model = smf.glm(
    formula='nuts_opened_count ~ age_years + C(sex) + received_help',
    data=_df,
    family=sm.families.Poisson(),
    offset=_df['log_seconds']
).fit()

print('N rows used:', len(_df))
print(_df[['age_years','sex','received_help','nuts_opened_count','session_seconds','efficiency']].head())
print(model.summary())

# Save key stats for later use if needed
coef = model.params
pvals = model.pvalues
results = pd.DataFrame({'coef': coef, 'pval': pvals})
results.to_csv('model_results.csv')
