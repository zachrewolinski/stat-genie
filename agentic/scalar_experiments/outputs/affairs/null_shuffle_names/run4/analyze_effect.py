import pandas as pd
import numpy as np

# Load data

df = pd.read_csv('affairs.csv')

# Map columns: based on value patterns
# 'religiousness' is yes/no -> likely children indicator
# 'age' has values {0,1,2,3,7,12} -> likely affairs frequency category
children_col = 'religiousness'
affairs_col = 'age'

# Create binary children indicator: yes=1, no=0
children = df[children_col].map({'yes': 1, 'no': 0})

# Outcome
affairs = df[affairs_col]

# Basic stats
mean_yes = affairs[children == 1].mean()
mean_no = affairs[children == 0].mean()

# Proportion with any affairs (>0)
prop_yes = (affairs[children == 1] > 0).mean()
prop_no = (affairs[children == 0] > 0).mean()

# Simple effect size (Cohen's d)
std_pooled = np.sqrt(((affairs[children == 1].var(ddof=1) + affairs[children == 0].var(ddof=1)) / 2))
cohen_d = (mean_no - mean_yes) / std_pooled if std_pooled > 0 else np.nan

print('mean_affairs_children_yes', mean_yes)
print('mean_affairs_children_no', mean_no)
print('diff_no_minus_yes', mean_no - mean_yes)
print('prop_any_children_yes', prop_yes)
print('prop_any_children_no', prop_no)
print('prop_diff_no_minus_yes', prop_no - prop_yes)
print('cohen_d_no_minus_yes', cohen_d)

# Also compare median
print('median_yes', affairs[children == 1].median())
print('median_no', affairs[children == 0].median())

# sample sizes
print('n_yes', (children==1).sum())
print('n_no', (children==0).sum())
