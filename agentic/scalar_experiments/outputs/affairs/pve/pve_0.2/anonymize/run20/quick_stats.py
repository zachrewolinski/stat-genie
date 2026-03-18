import pandas as pd
import numpy as np

df = pd.read_csv('affairs.csv')
child_yes = df['feature6'].astype(str).str.lower().eq('yes')
any_affair = df['feature2'] > 0

prop_yes = any_affair[child_yes].mean()
prop_no = any_affair[~child_yes].mean()

print('any_affair_prop_yes', prop_yes)
print('any_affair_prop_no', prop_no)

# effect size for difference in means
mean_yes = df.loc[child_yes, 'feature2'].mean()
mean_no = df.loc[~child_yes, 'feature2'].mean()
print('mean_yes', mean_yes)
print('mean_no', mean_no)

# difference
print('mean_diff_yes_minus_no', mean_yes - mean_no)
