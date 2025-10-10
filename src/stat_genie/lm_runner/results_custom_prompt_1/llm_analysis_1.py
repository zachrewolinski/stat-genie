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
# Drop rows with missing values in relevant columns
df = df.dropna(subset=['masfem', 'alldeaths', 'gender_mf'])


# ======== MODEL CODE ========
model = smf.ols('alldeaths ~ gender_mf * masfem', data=df).fit()
# Display the regression results
print(model.summary())


