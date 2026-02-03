import pandas as pd
import numpy as np
from statsmodels.stats.weightstats import ttest_ind
import statsmodels.api as sm

# Load data
path = "affairs.csv"
df = pd.read_csv(path)

# Based on value ranges and metadata mismatches, interpret columns:
# - 'age' holds affair frequency count (0-12)
# - 'religiousness' is a yes/no indicator that aligns with having children
# We'll analyze whether having children (yes/no) is associated with lower affair frequency.

# Prepare variables
children_yes = df["religiousness"].str.lower().map({"yes": 1, "no": 0})
affairs_count = df["age"].astype(float)

# Basic group stats
stats = df.groupby(children_yes).agg(
    mean_affairs=("age", "mean"),
    median_affairs=("age", "median"),
    prop_affair_gt0=("age", lambda s: (s > 0).mean()),
    n=("age", "size"),
)

# T-test for difference in means
x = affairs_count[children_yes == 1]
y = affairs_count[children_yes == 0]
t_stat, p_value, dfree = ttest_ind(x, y, usevar="unequal")

# Simple regression: affairs count ~ children indicator
X = sm.add_constant(children_yes)
model = sm.OLS(affairs_count, X).fit()

print("Group stats by children (0=no, 1=yes):")
print(stats)
print("\nT-test (children yes vs no):")
print(f"t={t_stat:.3f}, p={p_value:.4f}")
print("\nOLS: affairs_count ~ children_yes")
print(model.summary().tables[1])
