import pandas as pd
import numpy as np

df = pd.read_csv('crofoot.csv')
# Check if f_other equals dist_focal + other
print('f_other == dist_focal + other:', (df['f_other'] == df['dist_focal'] + df['other']).all())
print('win == dist_focal + other:', (df['win'] == df['dist_focal'] + df['other']).all())
print('f_other == focal + f_focal:', (df['f_other'] == df['focal'] + df['f_focal']).all())
print('win == focal + f_focal:', (df['win'] == df['focal'] + df['f_focal']).all())

# show mismatch counts
for label, cond in {
    'f_other == dist_focal + other': df['f_other'] == df['dist_focal'] + df['other'],
    'win == dist_focal + other': df['win'] == df['dist_focal'] + df['other'],
    'f_other == focal + f_focal': df['f_other'] == df['focal'] + df['f_focal'],
    'win == focal + f_focal': df['win'] == df['focal'] + df['f_focal'],
}.items():
    print(label, cond.sum(), 'of', len(cond))

