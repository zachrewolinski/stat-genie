import pandas as pd
import numpy as np

df = pd.read_csv('amtl.csv')
nearest = df['num_amtl'].round()
print('mean abs diff from nearest int', (df['num_amtl']-nearest).abs().mean())
print('min abs diff', (df['num_amtl']-nearest).abs().min())
print('max abs diff', (df['num_amtl']-nearest).abs().max())
print('fraction within 0.05 of int', ((df['num_amtl']-nearest).abs() < 0.05).mean())
