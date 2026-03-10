import pandas as pd

_df = pd.read_csv('crofoot.csv')

pairs = [
    ('dist_focal','other'),
    ('dist_focal','f_focal'),
    ('focal','other'),
    ('focal','f_focal'),
]

for total_col in ['f_other','win']:
    print('Checking total_col', total_col)
    for a,b in pairs:
        matches = (_df[a] + _df[b] == _df[total_col]).mean()
        print(f'  {a}+{b} matches: {matches:.2f}')
