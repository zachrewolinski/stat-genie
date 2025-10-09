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
# Calculate the average perception of threat level for each hurricane based on the total deaths caused
df['alldeaths'] = df['alldeaths'].astype(float)

# Drop any rows with missing values in the relevant columns
relevant_cols = ['masfem', 'alldeaths', 'gender_mf']
df = df.dropna(subset=relevant_cols)


# ======== MODEL CODE ========
model = smf.ols('alldeaths ~ gender_mf * masfem', data=df).fit()
# Display the regression results
print(model.summary())


