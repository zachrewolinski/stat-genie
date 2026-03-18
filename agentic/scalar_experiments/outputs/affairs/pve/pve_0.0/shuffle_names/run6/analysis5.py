import pandas as pd
import numpy as np
from scipy import stats


df = pd.read_csv('affairs.csv')
children = df['religiousness'].map({'yes': 1, 'no': 0})

numeric_cols = df.select_dtypes(include=[np.number]).columns

results = []
for col in numeric_cols:
    no_group = df[col][children == 0]
    yes_group = df[col][children == 1]
    t_stat, p_value = stats.ttest_ind(no_group, yes_group, equal_var=False, nan_policy='omit')
    mean_diff = no_group.mean() - yes_group.mean()
    pooled_sd = np.sqrt(((no_group.std(ddof=1) ** 2) + (yes_group.std(ddof=1) ** 2)) / 2)
    d = mean_diff / pooled_sd if pooled_sd != 0 else np.nan
    results.append((col, mean_diff, d, p_value))

results_sorted = sorted(results, key=lambda x: x[3])
for col, mean_diff, d, p_value in results_sorted:
    print(col, 'mean_diff', mean_diff, 'd', d, 'p', p_value)
