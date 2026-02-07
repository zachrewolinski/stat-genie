import pandas as pd
import numpy as np

# Load data
_df = pd.read_csv('affairs.csv')

# Map columns based on metadata descriptions (names are shuffled)
# children indicator appears in column 'religiousness' (yes/no)
# affairs frequency appears in column 'age' (0,1,2,3,7,12)
children_col = 'religiousness'
affairs_col = 'age'

# Basic cleaning
# Ensure binary children flag
children = _df[children_col].astype(str).str.lower()

# Some datasets might include 'yes'/'no' only
# Keep rows with yes/no
mask = children.isin(['yes','no'])

df = _df[mask].copy()
df['children_yes'] = (children[mask] == 'yes')
df['affairs'] = pd.to_numeric(df[affairs_col], errors='coerce')

# Remove missing affairs
df = df.dropna(subset=['affairs'])

# Compute mean affairs and proportion with any affairs
summary = df.groupby('children_yes')['affairs'].agg(['mean','median','count'])
summary['prop_any_affair'] = df.groupby('children_yes')['affairs'].apply(lambda s: (s > 0).mean())

# Effect sizes
mean_no = summary.loc[False, 'mean']
mean_yes = summary.loc[True, 'mean']
prop_no = summary.loc[False, 'prop_any_affair']
prop_yes = summary.loc[True, 'prop_any_affair']

# Differences (children yes - no)
mean_diff = mean_yes - mean_no
prop_diff = prop_yes - prop_no

# Simple standardized effect (Cohen's d) for affairs counts
# Using pooled std
std_no = df.loc[df['children_yes']==False, 'affairs'].std(ddof=1)
std_yes = df.loc[df['children_yes']==True, 'affairs'].std(ddof=1)

n_no = summary.loc[False, 'count']
n_yes = summary.loc[True, 'count']

pooled_std = np.sqrt(((n_no-1)*std_no**2 + (n_yes-1)*std_yes**2) / (n_no + n_yes - 2))
cohen_d = (mean_yes - mean_no) / pooled_std if pooled_std and not np.isnan(pooled_std) else np.nan

print('Summary by children_yes (False=no children, True=children):')
print(summary)
print('\nMean diff (yes - no):', mean_diff)
print('Prop any affair diff (yes - no):', prop_diff)
print('Cohen d:', cohen_d)
