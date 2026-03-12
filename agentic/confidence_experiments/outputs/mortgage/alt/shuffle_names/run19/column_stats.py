import pandas as pd, json

df = pd.read_csv('mortgage.csv')
info = {}
for col in df.columns:
    s = df[col]
    info[col] = {
        'mean': float(s.mean()),
        'std': float(s.std()),
        'min': float(s.min()),
        'max': float(s.max()),
        'unique': int(s.nunique()),
    }

print(json.dumps(info, indent=2, sort_keys=True))
