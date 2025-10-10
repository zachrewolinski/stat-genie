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
# Convert categorical gender_mf column to reflect feminity perception
# A higher value indicates a more feminine perception
# For binary column, 0 represents male and 1 represents female
df['FemininityRating'] = 11 - df['gender_mf'] * 10

# Define PrecautionaryMeasures as a function of FemininityRating
# Higher perceived femininity implies fewer precautionary measures
# Lower perceived femininity implies more precautionary measures
# This is a hypothetical relationship for illustrative purposes
df['PrecautionaryMeasures'] = 10 - df['FemininityRating']

# Drop any unnecessary columns
# In this case, we may drop columns not used in the statistical model
# df = df.drop(['ind', 'year', 'name', 'min', 'alldeaths', 'ndam', 'elapsedyrs', 'source', 'masfem_mturk', 'wind', 'ndam15'], axis=1)


# ======== MODEL CODE ========
model = smf.ols('PrecautionaryMeasures ~ HurricaneCategory * FemininityRating', data=df).fit()
# Display the regression results
print(model.summary())


