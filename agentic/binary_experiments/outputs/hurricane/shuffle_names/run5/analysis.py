import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

# Load data
_df = pd.read_csv('hurricane.csv')

# Map columns based on observed values
# year of hurricane
col_year = 'wind'
# hurricane name (string)
col_name = 'alldeaths'
# femininity rating (1-11 scale)
col_fem = 'category'
# min pressure at landfall
col_pressure = 'ndam15'
# binary female indicator
col_female_bin = 'masfem_mturk'
# Saffir-Simpson category
col_saffir = 'gender_mf'
# total deaths
col_deaths = 'name'
# max wind speed at landfall
col_maxwind = 'year'

# Prepare analysis frame
_df = _df.copy()
_df['log_deaths'] = np.log1p(_df[col_deaths])

# Core model: femininity rating with controls
formula_core = (
    'log_deaths ~ {fem} + {maxwind} + {pressure} + {saffir} + {year}'
).format(fem=col_fem, maxwind=col_maxwind, pressure=col_pressure, saffir=col_saffir, year=col_year)

model_core = smf.ols(formula_core, data=_df).fit()

# Alternative model: binary female indicator with same controls
formula_bin = (
    'log_deaths ~ {female_bin} + {maxwind} + {pressure} + {saffir} + {year}'
).format(female_bin=col_female_bin, maxwind=col_maxwind, pressure=col_pressure, saffir=col_saffir, year=col_year)

model_bin = smf.ols(formula_bin, data=_df).fit()

# Uncontrolled association
model_unadj = smf.ols('log_deaths ~ {fem}'.format(fem=col_fem), data=_df).fit()

# Print key results for inspection
print('Core model (fem rating):')
print(model_core.summary().tables[1])
print('\nBinary model (female indicator):')
print(model_bin.summary().tables[1])
print('\nUnadjusted (fem rating only):')
print(model_unadj.summary().tables[1])
