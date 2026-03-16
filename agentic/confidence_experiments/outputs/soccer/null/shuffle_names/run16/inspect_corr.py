import pandas as pd
import numpy as np

path='soccer.csv'
df=pd.read_csv(path)

cols=['yellowCards','meanExp','yellowReds','redCards']
print(df[cols].corr())
