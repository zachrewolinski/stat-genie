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
# No specific data transformations required for this analysis as the dataset already contains necessary columns
# The analysis will focus on the relationship between femininity of hurricane names, gender of the name, and the precautionary measures taken by the general public.


# ======== MODEL CODE ========
model = smf.ols('alldeaths ~ masfem * gender_mf', data=df).fit()
# Display the regression results
print(model.summary())


