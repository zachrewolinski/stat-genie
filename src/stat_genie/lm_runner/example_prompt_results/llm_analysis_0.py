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
# Convert 'gender_mf' column to indicate gender perception (0 for male, 1 for female)
df['GenderPerception'] = df['gender_mf']

# Create a new column to indicate whether precautionary measures were taken (1 if yes, 0 if no)
df['PrecautionaryMeasures'] = df['alldeaths'].apply(lambda x: 0 if x == 0 else 1)


# ======== MODEL CODE ========
model = smf.logit('PrecautionaryMeasures ~ GenderPerception * WindSpeed', data=df).fit()
# Display the logistic regression results
print(model.summary())


