from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/projects/binyu/hao_huang/stat-genie/blade/blade_bench/datasets/hurricane/data.csv')

# ======== TRANSFORM CODE ========
# Filter out hurricanes with missing masculinity-femininity (masfem) ratings
df = df.dropna(subset=['masfem'])

# Create a binary indicator for feminine hurricane names (1 for feminine, 0 for masculine)
df['IsFeminine'] = df['gender_mf']

# Standardize the masculinity-femininity ratings for computational ease
min_masfem = df['masfem'].min()
max_masfem = df['masfem'].max()
df['masfem'] = (df['masfem'] - min_masfem) / (max_masfem - min_masfem)

# Define a new variable 'PerceivedThreatLevel' based on the masculinity-femininity of hurricane names
# Higher masculinity-femininity ratings indicate less perceived threat
# We reverse the scale to align with the research hypothesis
# Higher values mean more feminine names (less threat)
df['PerceivedThreatLevel'] = 1 - df['masfem']


# ======== MODEL CODE ========
model = smf.ols('PerceivedThreatLevel ~ IsFeminine * masfem', data=df).fit()
# Display the regression results
print(model.summary())


