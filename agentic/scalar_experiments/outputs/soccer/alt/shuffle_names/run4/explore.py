import pandas as pd
import numpy as np

path = '/home/chenwang/stat-genie/agentic/scalar_experiments/outputs/soccer/alt/shuffle_names/run4/soccer.csv'

df = pd.read_csv(path)
print(df.head())
print('cols', df.columns)
# numeric summary
num_cols = df.select_dtypes(include=[np.number]).columns
summary = []
for c in num_cols:
    s = df[c]
    summary.append((c, s.min(), s.max(), s.mean(), s.std(), s.nunique()))
summary = sorted(summary, key=lambda x: x[1])
print('numeric summary:')
for row in summary:
    print(row)

cand_skin = [c for c in num_cols if df[c].min()>=0 and df[c].max()<=1 and df[c].nunique()<=6]
print('candidate skin columns', cand_skin)

cand_small = [c for c in num_cols if df[c].max()<=5 and df[c].min()>=0 and df[c].nunique()<=6]
print('candidate small count columns', cand_small)
