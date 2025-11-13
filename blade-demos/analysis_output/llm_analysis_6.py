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
# No missing values in the dataset

df['AvgPerceivedThreat'] = df['alldeaths']



# ======== MODEL CODE ========
model = smf.ols('AvgPerceivedThreat ~ gender_mf * masfem', data=df).fit()
# Display the regression results
print(model.summary())


