import pandas as pd
import itertools

raw = pd.read_csv('amtl.csv')
num_cols = ['genus','age','pop','num_amtl','stdev_age']

# Only integer-like columns for counts? but pop/num_amtl floats maybe counts too.

for a,b in itertools.permutations(num_cols,2):
    # check if all a<=b (possible missing<=sockets) if both non-negative
    if (raw[a] >= 0).all() and (raw[b] >= 0).all():
        if (raw[a] <= raw[b]).all():
            print(f"{a} <= {b}")

