import pandas as pd
import numpy as np

df = pd.read_csv('panda_nuts.csv')
df['efficiency'] = df['nuts_opened'] / df['seconds']

summary = {
    'n_total': len(df),
    'n_help_yes': int((df['help']=='y').sum()),
    'n_help_no': int((df['help']=='N').sum()),
    'eff_mean': df['efficiency'].mean(),
    'eff_sd': df['efficiency'].std(),
    'eff_median': df['efficiency'].median(),
    'eff_by_sex': df.groupby('sex')['efficiency'].mean().to_dict(),
    'eff_by_help': df.groupby('help')['efficiency'].mean().to_dict(),
    'age_mean': df['age'].mean(),
    'age_sd': df['age'].std(),
}

for k,v in summary.items():
    print(k, v)

