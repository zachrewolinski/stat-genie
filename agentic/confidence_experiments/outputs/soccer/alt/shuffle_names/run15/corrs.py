import pandas as pd
import numpy as np


df = pd.read_csv('soccer.csv')

# candidates
cand1 = df['meanExp']
cand2 = df['yellowCards']
# proxy for yellow cards
yellow = df['yellowReds']

def corr(a,b):
    return a.corr(b)

print('corr meanExp vs yellowReds:', corr(cand1, yellow))
print('corr yellowCards vs yellowReds:', corr(cand2, yellow))

# Spearman
print('spearman meanExp vs yellowReds:', cand1.corr(yellow, method='spearman'))
print('spearman yellowCards vs yellowReds:', cand2.corr(yellow, method='spearman'))
