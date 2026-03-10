import pandas as pd
import numpy as np

df = pd.read_csv('hurricane.csv')
num_cols = df.select_dtypes(include=['number']).columns
corr = df[num_cols].corr()
# find highest correlations for masfem
print(corr['masfem'].sort_values(ascending=False))
