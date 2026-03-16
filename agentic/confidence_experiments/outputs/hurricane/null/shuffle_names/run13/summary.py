import pandas as pd
import numpy as np

_df = pd.read_csv('hurricane.csv')
_df['name'] = pd.to_numeric(_df['name'], errors='coerce')
_df['masfem_mturk'] = pd.to_numeric(_df['masfem_mturk'], errors='coerce')
_df['category'] = pd.to_numeric(_df['category'], errors='coerce')
_df['ind'] = pd.to_numeric(_df['ind'], errors='coerce')

# group by binary gender
summary = _df.groupby('masfem_mturk')['name'].agg(['count','mean','median'])
print(summary)

# also show log deaths mean
summary_log = _df.assign(log_deaths=np.log1p(_df['name'])).groupby('masfem_mturk')['log_deaths'].agg(['mean','median'])
print(summary_log)

