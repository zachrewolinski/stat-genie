import pandas as pd
import numpy as np

df = pd.read_csv('amtl.csv')
mult = df['feature3'] * df['feature4']
diff = np.abs(mult - np.round(mult))
print(diff.describe())
print('max diff', diff.max())
print('mean diff', diff.mean())
print('unique diffs top', np.unique(np.round(diff,3))[:10])
