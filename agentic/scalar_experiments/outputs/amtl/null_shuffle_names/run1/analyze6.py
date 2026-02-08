import pandas as pd
import numpy as np

df = pd.read_csv('amtl.csv')
num = df['num_amtl']
near_int = (num - num.round()).abs() < 1e-6
print('near int %', near_int.mean())
print('mean abs diff to nearest int', (num - num.round()).abs().mean())
print('sample diffs', (num - num.round()).abs().head())
