import pandas as pd
import numpy as np

path = 'soccer.csv'
df = pd.read_csv(path)

use = df[['photoID','rater1','nExp','redCards','yellowCards']].dropna().copy()
use['skin_avg'] = (use['rater1'] + use['nExp']) / 2

# unique values and counts
counts = use['skin_avg'].value_counts().sort_index()
print('skin_avg value counts:')
print(counts)

# unique values for rater1 and nExp
print('rater1 counts:', use['rater1'].value_counts().sort_index())
print('nExp counts:', use['nExp'].value_counts().sort_index())

# player-level unique skin_avg values
player_skin = use.groupby('photoID')['skin_avg'].mean().value_counts().sort_index()
print('player-level skin_avg counts:')
print(player_skin)

