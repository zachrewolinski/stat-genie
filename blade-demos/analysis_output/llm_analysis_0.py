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
# No transformations needed as the data is already clean and in the required format


# ======== MODEL CODE ========
model = smf.ols('PrecautionaryMeasures ~ IsFeminineName * masfem', data=df).fit()
# Display the regression results
print(model.summary())


