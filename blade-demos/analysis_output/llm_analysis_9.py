from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/blade/blade_bench/datasets/hurricane/data.csv')

# ======== TRANSFORM CODE ========
# No missing values handling needed for this analysis

# Create a binary variable to represent feminine names (1: feminine, 0: masculine)
df['gender_mf'] = df['gender_mf'].astype(int)

# Define the model variables
X = df[['masfem', 'gender_mf', 'masfem:gender_mf']]
y = df['alldeaths']


# ======== MODEL CODE ========
model = smf.ols('alldeaths ~ masfem * gender_mf', data=df).fit()
# Display the regression results
print(model.summary())


