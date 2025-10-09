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
# Convert categorical gender indicator to interpretable text
# Female if gender_mf = 1, Male if gender_mf = 0
df['Gender'] = df['gender_mf'].apply(lambda x: 'Female' if x == 1 else 'Male')

# Drop any rows with missing values in relevant columns
df = df.dropna(subset=['masfem', 'alldeaths', 'gender_mf'])



# ======== MODEL CODE ========
model = smf.ols('alldeaths ~ masfem * gender_mf', data=df).fit()
# Display the regression results
print(model.summary())


