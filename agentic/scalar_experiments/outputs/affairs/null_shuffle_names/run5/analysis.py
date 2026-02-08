import pandas as pd
import numpy as np

# Load data
csv_path = 'affairs.csv'
df = pd.read_csv(csv_path)

# Based on metadata descriptions, the columns appear shifted. Infer actual roles:
# - 'age' column has values 0..12 with discrete codes matching affair frequency -> affairs outcome
# - 'religiousness' column has yes/no and description says "Are there children in the marriage?" -> children indicator

outcome = df['age']
children = df['religiousness'].map({'yes': 1, 'no': 0})

# Basic counts
n_total = len(df)
children_counts = children.value_counts(dropna=False)

# Means
mean_affairs_children = outcome[children == 1].mean()
mean_affairs_nochildren = outcome[children == 0].mean()

# Difference
diff = mean_affairs_children - mean_affairs_nochildren

# Effect size: Cohen's d
std_pooled = np.sqrt(((outcome[children == 1].var(ddof=1)) + (outcome[children == 0].var(ddof=1))) / 2)
cohen_d = diff / std_pooled if std_pooled > 0 else np.nan

# Nonparametric: fraction with any affairs (>0)
any_affairs_children = (outcome[children == 1] > 0).mean()
any_affairs_nochildren = (outcome[children == 0] > 0).mean()

# Simple summary stats by group
summary = df.assign(children=children, affairs=outcome).groupby('children')['affairs'].agg(['count','mean','median','std'])

print('n_total', n_total)
print('children_counts')
print(children_counts)
print('summary')
print(summary)
print('mean_affairs_children', mean_affairs_children)
print('mean_affairs_nochildren', mean_affairs_nochildren)
print('diff', diff)
print('cohen_d', cohen_d)
print('any_affairs_children', any_affairs_children)
print('any_affairs_nochildren', any_affairs_nochildren)
