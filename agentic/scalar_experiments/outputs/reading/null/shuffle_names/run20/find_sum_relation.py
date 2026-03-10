import pandas as pd
import itertools

_df = pd.read_csv('reading.csv')

# numeric columns
num_cols = [c for c in _df.columns if pd.api.types.is_numeric_dtype(_df[c])]

results = []

for a, b, c in itertools.permutations(num_cols, 3):
    # check if a approx b + c
    diff = (_df[a] - (_df[b] + _df[c])).abs()
    mae = diff.mean()
    # normalize by mean of a to compare
    mean_a = _df[a].abs().mean() + 1e-9
    rel = mae / mean_a
    if rel < 0.05:  # 5% relative error
        results.append((rel, a, b, c, mae))

results.sort()
print('found', len(results))
for rel, a, b, c, mae in results[:20]:
    print(f'{a} ≈ {b}+{c}: rel {rel:.4f}, mae {mae:.2f}')
