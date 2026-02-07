import pandas as pd
import numpy as np

df = pd.read_csv('boxes.csv')

df['chosen_demo'] = df['feature1'].isin([2,3]).astype(int)
df['majority_choice'] = (df['feature1'] == 2).astype(int)

# by site
site_stats = df.groupby('feature5').agg(
    n=('feature1','size'),
    social_reliance=('chosen_demo','mean'),
    majority_rate=('majority_choice','mean')
)

# by age
age_stats = df.groupby('feature3').agg(
    n=('feature1','size'),
    social_reliance=('chosen_demo','mean'),
    majority_rate=('majority_choice','mean')
)

print('Site stats:\n', site_stats)
print('\nAge stats:\n', age_stats)
