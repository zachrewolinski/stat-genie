import pandas as pd
import numpy as np

path = 'mortgage.csv'

df = pd.read_csv(path)

# identify binary columns
binary_cols = [c for c in df.columns if df[c].dropna().nunique() == 2]

print('Binary columns:', binary_cols)

# compute correlations with continuous columns
continuous_cols = [c for c in df.columns if df[c].dropna().nunique() > 2]

for b in binary_cols:
    print('\nOutcome candidate:', b)
    # correlation with continuous vars
    corrs = {}
    for c in continuous_cols:
        # skip if constant or all nan
        if df[c].dropna().nunique() <= 1:
            continue
        # compute point-biserial correlation
        try:
            corr = df[[b,c]].corr().iloc[0,1]
        except Exception:
            corr = np.nan
        corrs[c] = corr
    # sort by abs corr
    for c, corr in sorted(corrs.items(), key=lambda kv: (abs(kv[1]) if kv[1] is not None else -1), reverse=True)[:5]:
        print(f'  {c}: {corr:.3f}')
