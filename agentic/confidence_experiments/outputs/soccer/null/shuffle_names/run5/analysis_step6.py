import pandas as pd
import numpy as np

_df = pd.read_csv('soccer.csv')

# For each column with many unique values, compute pct groups where rater1 is constant
results = []
for col in _df.columns:
    if col == 'rater1':
        continue
    # ignore columns with too few unique values
    if _df[col].nunique() < 50:
        continue
    nunique_per_group = _df.groupby(col)['rater1'].nunique()
    pct_const = (nunique_per_group == 1).mean()
    results.append((col, pct_const, nunique_per_group.mean(), nunique_per_group.max()))

for col, pct_const, mean_nu, max_nu in sorted(results, key=lambda x: -x[1])[:10]:
    print(col, 'pct_const', pct_const, 'mean_nunique', mean_nu, 'max_nunique', max_nu)

