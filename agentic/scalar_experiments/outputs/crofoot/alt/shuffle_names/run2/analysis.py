import pandas as pd
import numpy as np

# Load data
_df = pd.read_csv('crofoot.csv')
print('shape', _df.shape)

summary = []
for col in _df.columns:
    s = _df[col]
    summary.append({
        'col': col,
        'dtype': s.dtype,
        'min': s.min(),
        'max': s.max(),
        'mean': s.mean(),
        'std': s.std(),
        'n_unique': s.nunique(),
        'unique_vals': sorted(s.unique())[:10]
    })

summary_df = pd.DataFrame(summary)
print(summary_df.to_string(index=False))
