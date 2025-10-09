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
# Convert categorical gender to binary (0 for male, 1 for female)
df['GenderBinary'] = df['gender_mf']
# Create a perceived threat level based on masculinity-femininity index
# Higher masculinity indicates higher threat perception
df['PerceivedThreatLevel'] = 11 - df['masfem']


# ======== MODEL CODE ========
model = smf.ols('PerceivedThreatLevel ~ GenderBinary * masfem', data=df).fit()
# Display the regression results
print(model.summary())


