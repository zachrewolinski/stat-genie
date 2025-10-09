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
# No missing data handling needed for this analysis

df['PrecautionaryMeasures'] = df['alldeaths'] + df['ndam']
df['FeminineName'] = df['gender_mf']


# ======== MODEL CODE ========
model = smf.ols('PrecautionaryMeasures ~ FeminineName * masfem', data=df).fit()
# Display the regression results
print(model.summary())


