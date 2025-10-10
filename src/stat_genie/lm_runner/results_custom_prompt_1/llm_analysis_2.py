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
# No transformation needed for this analysis as all required columns are already in the dataset


# ======== MODEL CODE ========
model = smf.ols('alldeaths ~ masfem * gender_mf', data=df).fit()
# Display the regression results
print(model.summary())


