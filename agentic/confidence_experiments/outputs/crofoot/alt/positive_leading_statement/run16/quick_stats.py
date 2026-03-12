import pandas as pd


df = pd.read_csv('crofoot.csv')
size_diff = df['n_focal'] - df['n_other']
loc_diff = df['dist_other'] - df['dist_focal']

# Win rates by advantage
for name, series in [('size_diff', size_diff), ('loc_diff', loc_diff)]:
    win = df['win']
    advant = series > 0
    disadv = series < 0
    tie = series == 0
    print(name)
    print('advantage win rate', win[advant].mean(), 'n', advant.sum())
    print('disadvantage win rate', win[disadv].mean(), 'n', disadv.sum())
    print('tie win rate', win[tie].mean(), 'n', tie.sum())
    print()

