import pandas as pd

csv_path = 'crofoot.csv'

df = pd.read_csv(csv_path)
print('Shape:', df.shape)
print('\nHead:')
print(df.head())
print('\nWin value counts:')
print(df['win'].value_counts())

# create relative size and location variables for inspection
_df = df.copy()
_df['rel_size'] = _df['n_focal'] - _df['n_other']
_df['rel_males'] = _df['m_focal'] - _df['m_other']
_df['rel_females'] = _df['f_focal'] - _df['f_other']
_df['loc_advantage'] = _df['dist_other'] - _df['dist_focal']  # >0 means focal closer to its center

print('\nRelative size summary:')
print(_df['rel_size'].describe())
print('\nLocation advantage summary:')
print(_df['loc_advantage'].describe())

print('\nCross-tab win by rel_size sign:')
print(pd.crosstab((_df['rel_size'] > 0).map({True: 'focal_larger', False: 'focal_not_larger'}), _df['win']))

print('\nCross-tab win by location advantage sign:')
print(pd.crosstab((_df['loc_advantage'] > 0).map({True: 'focal_closer', False: 'focal_not_closer'}), _df['win']))
