import os, sys
cwd = os.getcwd()
sys.path = [p for p in sys.path if p not in ("", cwd)]
import pandas as pd
import numpy as np

cols = ['redCards','nIAT','defeats','rater2','player','yellowReds','meanExp','yellowCards']

df = pd.read_csv('soccer.csv')
# check correlations with redCards
print('corr with redCards (games?)')
for c in cols:
    if c=='redCards':
        continue
    print(c, df['redCards'].corr(df[c]))

# check if redCards approx sum of some 3 columns: victories/defeats/ties maybe
# compute unique counts of redCards minus candidate sums
candidates = cols.copy()

# try all triplets
best = []
for i in range(len(candidates)):
    for j in range(i+1,len(candidates)):
        for k in range(j+1,len(candidates)):
            a,b,c = candidates[i], candidates[j], candidates[k]
            diff = df['redCards'] - (df[a]+df[b]+df[c])
            # check if diff mostly zero
            zero_pct = (diff==0).mean()
            if zero_pct > 0.5:
                best.append((zero_pct, (a,b,c)))

best_sorted = sorted(best, reverse=True)[:10]
print('top triplets summing to redCards', best_sorted)

# check if any two columns sum to redCards
best2 = []
for i in range(len(candidates)):
    for j in range(i+1,len(candidates)):
        a,b = candidates[i], candidates[j]
        diff = df['redCards'] - (df[a]+df[b])
        zero_pct = (diff==0).mean()
        if zero_pct > 0.5:
            best2.append((zero_pct, (a,b)))
print('top pairs', sorted(best2, reverse=True)[:10])

