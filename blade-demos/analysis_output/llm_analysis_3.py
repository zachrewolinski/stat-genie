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
# Create a binary column to identify hurricanes with feminine names
feminine_names = ['Alice', 'Bella', 'Cindy', ...]  # list of feminine names

df['IsFeminineName'] = df['name'].apply(lambda x: 1 if x in feminine_names else 0)

# Calculate the femininity perception based on the masculinity-femininity index
# Masculine values are inversely related to femininity perception
# Higher masfem values indicate more feminine names
# Normalize the masfem values to be between 0 and 1
min_masfem = df['masfem'].min()
max_masfem = df['masfem'].max()
df['FemininityPerception'] = 1 - (df['masfem'] - min_masfem) / (max_masfem - min_masfem)

# Assume that precautionary measures increase with increasing category (higher severity)
# Normalize the category values to be between 0 and 1
min_category = df['category'].min()
max_category = df['category'].max()
df['Category'] = (df['category'] - min_category) / (max_category - min_category)

# Generate a random sample of precautionary measures taken by the general public
# This is a hypothetical column for demonstration
np.random.seed(42)
df['PrecautionaryMeasures'] = np.random.randint(0, 101, size=len(df))


# ======== MODEL CODE ========
model = smf.ols('PrecautionaryMeasures ~ Category * FemininityPerception', data=df).fit()
# Display the regression results
print(model.summary())


