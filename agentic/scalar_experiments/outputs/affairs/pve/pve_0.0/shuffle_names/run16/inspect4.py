import pandas as pd
import numpy as np


df = pd.read_csv('affairs.csv')
# encode categorical yes/no and gender

df_enc = df.copy()
for col in df_enc.columns:
    if df_enc[col].dtype == 'object':
        df_enc[col] = df_enc[col].astype('category').cat.codes

corr = df_enc.corr(numeric_only=True)
print(corr)

