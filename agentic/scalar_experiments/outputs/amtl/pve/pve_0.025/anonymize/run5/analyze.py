import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
csv_path = 'amtl.csv'
df = pd.read_csv(csv_path)
print(df.head())
print(df.dtypes)
print(df.describe(include='all'))
