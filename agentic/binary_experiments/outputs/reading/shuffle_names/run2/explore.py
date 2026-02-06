import pandas as pd
import numpy as np

df = pd.read_csv('reading.csv')
print(df.nunique())
print('\nunique values for candidate columns:')
for col in ['language','device','dyslexia','dyslexia_bin','correct_rate','img_width','reader_view','english_native','page_id']:
    print(col, df[col].unique()[:10])

print('\nsummary numeric:')
print(df.describe(include='number').T[['min','max','mean','std']])
