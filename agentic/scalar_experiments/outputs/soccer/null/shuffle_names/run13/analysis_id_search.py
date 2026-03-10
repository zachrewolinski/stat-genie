import pandas as pd

path = 'soccer.csv'
df = pd.read_csv(path)

# Check for duplicate column names
if df.columns.duplicated().any():
    print('Duplicate column names detected')

results = []
for col in df.columns:
    tmp = df[[col, 'rater1', 'nExp']].dropna()
    # handle potential duplicate columns by ensuring series
    series = tmp[col]
    if isinstance(series, pd.DataFrame):
        # skip if ambiguous
        continue
    uniq = series.nunique()
    if uniq < 2:
        continue
    r1_var = (tmp.groupby(col)['rater1'].nunique() > 1).mean()
    r2_var = (tmp.groupby(col)['nExp'].nunique() > 1).mean()
    results.append((col, uniq, r1_var, r2_var))

results = sorted(results, key=lambda x: (x[2]+x[3]))
print('Top 10 candidate ids with least variation in rater1/nExp:')
for r in results[:10]:
    print(r)

