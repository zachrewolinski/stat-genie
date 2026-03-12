import pandas as pd

path='soccer.csv'
df=pd.read_csv(path)
print(df.head())
print("Rows", len(df))

for col in df.columns:
    s=df[col]
    print("\n", col)
    print(s.dtype)
    print("sample", s.head(3).tolist())
    if pd.api.types.is_numeric_dtype(s):
        print("min", s.min(), "max", s.max(), "mean", s.mean(), "std", s.std(), "unique", s.nunique())
    else:
        print("unique", s.nunique())
