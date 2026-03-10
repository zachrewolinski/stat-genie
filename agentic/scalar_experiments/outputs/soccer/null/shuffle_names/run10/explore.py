import pandas as pd
import numpy as np

path='soccer.csv'

df=pd.read_csv(path)

summary=[]
for col in df.columns:
    s=df[col]
    if pd.api.types.is_numeric_dtype(s):
        summary.append({
            'col': col,
            'dtype': str(s.dtype),
            'min': float(np.nanmin(s)),
            'max': float(np.nanmax(s)),
            'mean': float(np.nanmean(s)),
            'std': float(np.nanstd(s)),
            'n_unique': int(s.nunique()),
        })
    else:
        summary.append({
            'col': col,
            'dtype': 'object',
            'n_unique': int(s.nunique()),
            'sample': s.dropna().astype(str).head(3).tolist(),
        })

summary_df=pd.DataFrame(summary)
print(summary_df)

# identify candidate skin tone columns
candidates=[]
for col in df.columns:
    s=df[col]
    if pd.api.types.is_numeric_dtype(s):
        uniq=sorted(s.dropna().unique())
        # candidate if values within [0,1] and roughly increments of 0.25 or 0.5
        if s.min()>=0 and s.max()<=1 and s.nunique()<=6:
            candidates.append((col, s.nunique(), uniq[:10]))
print('\nCandidates 0-1 with few uniques:')
for c in candidates:
    print(c)

# identify candidate red card columns: integer counts with many zeros and small max (<=10 maybe)
rc_candidates=[]
for col in df.columns:
    s=df[col]
    if pd.api.types.is_numeric_dtype(s):
        if (s.dropna()>=0).all() and (s.dropna()%1==0).all():
            # integer
            maxv=s.max()
            if maxv<=10:
                rc_candidates.append((col, int(maxv), int((s==0).mean()*100)))
print('\nInteger count columns with max<=10:')
for c in rc_candidates:
    print(c)

