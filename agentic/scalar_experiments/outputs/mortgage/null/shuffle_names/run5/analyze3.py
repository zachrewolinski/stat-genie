import pandas as pd

_df = pd.read_csv('mortgage.csv')

# Check complement relations
pairs = [('self_employed','deny'), ('self_employed','accept'), ('deny','accept')]
for a,b in pairs:
    if a in _df.columns and b in _df.columns:
        comp = ((_df[a] + _df[b])==1).mean()
        print(a,b,'share sum==1',comp)
        print('unique pairs', _df[[a,b]].drop_duplicates().sort_values([a,b]).head())

# Check if deny equals 1- self_employed
if 'self_employed' in _df.columns and 'deny' in _df.columns:
    diff = (_df['deny'] - (1-_df['self_employed'])).abs().mean()
    print('mean abs diff deny vs 1-self_employed', diff)
