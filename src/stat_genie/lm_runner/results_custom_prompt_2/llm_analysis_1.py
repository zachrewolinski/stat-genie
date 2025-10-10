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
# No missing values to handle in this dataset

# Define a binary column for feminine names
# Assumption: Masculine names are coded as 0, and feminine names are coded as 1
# This is based on the gender_mf column

df['FeminineName'] = df['gender_mf']

# Filter out rows where alldeaths field is missing
# Assumption: Precautionary measures can be deduced from the number of deaths caused by the hurricane

# Convert masculine-feminine index to a categorical variable
# Assumption: If masfem is above the median, the name is considered feminine, otherwise masculine
median_masfem = df['masfem'].median()
df['MasculineFeminine'] = df['masfem'].apply(lambda x: 1 if x > median_masfem else 0)

# ======== MODEL CODE ========
model = smf.ols('alldeaths ~ MasculineFeminine + MasculineFeminine * FeminineName', data=df).fit()
print(model.summary())

