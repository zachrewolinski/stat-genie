import pandas as pd
import numpy as np
from scipy import stats

df = pd.read_csv("affairs.csv")

# Map columns based on value patterns
# affairs frequency is in column with values {0,1,2,3,7,12}
affairs_col = None
for col in df.columns:
    vals = set(df[col].dropna().unique())
    if vals.issubset({0, 1, 2, 3, 7, 12}):
        affairs_col = col
        break
if affairs_col is None:
    raise RuntimeError('Could not identify affairs column')

# children indicator is in yes/no column
children_col = None
for col in df.columns:
    if df[col].dtype == object:
        vals = set(df[col].dropna().unique())
        if vals.issubset({'yes', 'no'}):
            children_col = col
            break
if children_col is None:
    raise RuntimeError('Could not identify children column')

# Prepare data
analysis_df = df[[affairs_col, children_col]].copy()
analysis_df['children'] = analysis_df[children_col].map({'yes': 1, 'no': 0})
analysis_df['affairs'] = analysis_df[affairs_col]

# Basic summaries
summary = analysis_df.groupby('children')['affairs'].agg(['count', 'mean', 'median'])

# Proportion with any affair
analysis_df['any_affair'] = (analysis_df['affairs'] > 0).astype(int)
prop_any = analysis_df.groupby('children')['any_affair'].mean()

# Two-sample tests (nonparametric for counts)
with_children = analysis_df[analysis_df['children'] == 1]['affairs']
without_children = analysis_df[analysis_df['children'] == 0]['affairs']

# Mann-Whitney U test
u_stat, u_p = stats.mannwhitneyu(with_children, without_children, alternative='two-sided')

# Difference in means and Cohen's d
mean_diff = with_children.mean() - without_children.mean()
pooled_std = np.sqrt(((with_children.std(ddof=1) ** 2) + (without_children.std(ddof=1) ** 2)) / 2)
cohens_d = mean_diff / pooled_std if pooled_std != 0 else np.nan

print('affairs_col:', affairs_col)
print('children_col:', children_col)
print('summary:\n', summary)
print('prop_any:\n', prop_any)
print('mean_diff (with - without):', mean_diff)
print('cohens_d:', cohens_d)
print('mannwhitney_u_p:', u_p)
