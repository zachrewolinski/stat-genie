import pandas as pd
import numpy as np

_df = pd.read_csv('soccer.csv')

# candidate red columns (rare counts)
red_candidates = ['meanExp', 'yellowCards']

# candidate exposure columns: integer counts, min>=1, zero proportion 0, max <= 200
exp_candidates = []
for col in _df.columns:
    if pd.api.types.is_numeric_dtype(_df[col]):
        s = _df[col].dropna()
        if len(s)==0:
            continue
        if np.allclose(s % 1, 0) and s.min() >= 1 and (s==0).mean()==0 and s.max() <= 200:
            exp_candidates.append(col)

print('exp_candidates', exp_candidates)

for exp_col in exp_candidates:
    for red_col in red_candidates:
        corr = np.corrcoef(_df[exp_col], _df[red_col])[0,1]
        print('corr', exp_col, red_col, corr)
