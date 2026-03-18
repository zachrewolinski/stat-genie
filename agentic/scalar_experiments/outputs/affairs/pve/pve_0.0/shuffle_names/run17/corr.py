import pandas as pd
import numpy as np

df = pd.read_csv('affairs.csv')
num_cols = df.select_dtypes(include=[np.number]).columns

corr = df[num_cols].corr()
print(corr['age'].sort_values())

# check correlation of education with others
print('\nEducation correlations:')
print(corr['education'].sort_values())
