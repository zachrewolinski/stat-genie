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
# Drop rows with missing values
# Convert categorical variable 'name' to binary indicator for gender
# Define a moderator variable based on masculinity-femininity index
df = df.dropna(subset=['masfem'])
df['GenderMF'] = df['name'].apply(lambda x: 1 if x[-1].lower() == 'a' else 0)
df['IsFeminineName'] = df['GenderMF']
df['InFemaleCategory'] = df['name'].apply(lambda x: 1 if x[-1].lower() in ['a', 'e'] else 0)


# ======== MODEL CODE ========
model = smf.ols('alldeaths ~ GenderMF * masfem', data=df).fit()
# Display the regression results
print(model.summary())


