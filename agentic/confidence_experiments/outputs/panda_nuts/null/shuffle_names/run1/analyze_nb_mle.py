import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

path = 'panda_nuts.csv'
df = pd.read_csv(path)

analysis = pd.DataFrame({
    'age_years': df['hammer'].astype(float),
    'sex': df['nuts_opened'].astype(str),
    'help_received': df['seconds'].astype(str),
    'nuts_opened': df['help'].astype(float),
    'duration_sec': df['chimpanzee'].astype(float),
})

analysis['log_duration'] = np.log(analysis['duration_sec'])

# Use discrete NegativeBinomial to estimate alpha
nb2 = smf.negativebinomial(
    formula='nuts_opened ~ age_years + C(sex) + C(help_received)',
    data=analysis,
    exposure=analysis['duration_sec']
).fit(disp=False)

print(nb2.summary())

print('alpha', nb2.params.get('alpha'))
