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
# No transformation needed in this case as the data is already clean and ready for modeling.


# ======== MODEL CODE ========
model = smf.ols('alldeaths ~ gender_mf * masfem', data=df).fit()
# Display the regression results
print(model.summary())


