import pandas as pd
import numpy as np
from scipy import stats

# Load data

df = pd.read_csv('reading.csv')
print(df.head())
print(df.describe(include='all'))
