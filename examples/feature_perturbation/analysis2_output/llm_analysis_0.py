from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/.venv/lib/python3.10/site-packages/blade_bench/datasets/hurricane/data.csv')

# ======== TRANSFORM CODE ========
# Copy dataframe to avoid modifying in place
_df = df.copy()

# Rename columns to meaningful names used below
_df = _df.rename(columns={
    'feature6': 'Name',
    'feature12': 'FeminineName',          # binary (0 male, 1 female)
    'feature9': 'NameFemininity',        # continuous rater index (higher = more feminine)
    'feature13': 'Deaths',               # fatalities (count)
    'feature7': 'MaxWind',               # maximum wind speed at landfall
    'feature4': 'Category',              # Saffir-Simpson category
    'feature14': 'MinPressure',          # minimum central pressure
    'feature5': 'Year',                  # year of storm
    'feature8': 'Damage2015',            # damage normalized to 2015
    'feature10': 'Source',               # data source (text)
    'feature2': 'StormID',
    'feature11': 'MTurkRating',
    'feature1': 'Damage2013',
    'feature3': 'YearsSince'
})

# Ensure numeric columns are numeric; coerce to NaN on errors
num_cols = ['FeminineName', 'NameFemininity', 'Deaths', 'MaxWind', 'Category', 'MinPressure', 'Year', 'Damage2015']
for c in num_cols:
    _df[c] = pd.to_numeric(_df[c], errors='coerce')

# Source as categorical/string
_df['Source'] = _df['Source'].astype(str)

# Drop rows missing critical variables for this analysis
_df = _df.dropna(subset=['Deaths', 'NameFemininity', 'FeminineName', 'MaxWind', 'Category', 'MinPressure', 'Year', 'Damage2015'])

# Create log-transformed damage to reduce skew (add 1 to avoid log(0))
_df['LogDamage'] = np.log(_df['Damage2015'] + 1)

# Standardize continuous controls and IV (z-score) to aid interpretation and model convergence
_df['NameFemininity_z'] = (_df['NameFemininity'] - _df['NameFemininity'].mean()) / (_df['NameFemininity'].std(ddof=0) if _df['NameFemininity'].std(ddof=0) != 0 else 1)
_df['MaxWind_z'] = (_df['MaxWind'] - _df['MaxWind'].mean()) / (_df['MaxWind'].std(ddof=0) if _df['MaxWind'].std(ddof=0) != 0 else 1)
_df['Category_z'] = (_df['Category'] - _df['Category'].mean()) / (_df['Category'].std(ddof=0) if _df['Category'].std(ddof=0) != 0 else 1)
_df['MinPressure_z'] = (_df['MinPressure'] - _df['MinPressure'].mean()) / (_df['MinPressure'].std(ddof=0) if _df['MinPressure'].std(ddof=0) != 0 else 1)

# Center year (linear trend control)
_df['Year_c'] = _df['Year'] - _df['Year'].mean()

# Keep only the columns required for modeling (plus identifiers)
final_cols = [
    'StormID', 'Name', 'FeminineName', 'NameFemininity', 'NameFemininity_z', 'Deaths',
    'MaxWind', 'MaxWind_z', 'Category', 'Category_z', 'MinPressure', 'MinPressure_z',
    'Year', 'Year_c', 'Damage2015', 'LogDamage', 'Source', 'MTurkRating'
]
_df = _df.loc[:, [c for c in final_cols if c in _df.columns]]

# Return transformed dataframe
df = _df
return df

# ======== MODEL CODE ========
import statsmodels.formula.api as smf
import statsmodels.api as sm

# Fit a negative binomial regression predicting deaths from name femininity (binary and continuous)
# while controlling for measures of storm severity and exposure. We include Source as a categorical control.
# Using a count model (negative binomial) is appropriate for overdispersed count outcomes like fatalities.
formula = 'Deaths ~ FeminineName + NameFemininity_z + MaxWind_z + Category_z + MinPressure_z + Year_c + LogDamage + C(Source)'

model = smf.glm(formula=formula, data=df, family=sm.families.NegativeBinomial())
results = model.fit()

# Provide summary for inspection and return fitted results object
print(results.summary())
return results

