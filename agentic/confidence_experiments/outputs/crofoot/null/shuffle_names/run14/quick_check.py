import pandas as pd

df = pd.read_csv('crofoot.csv')

# check mapping via sum males+females
checks = {
    'f_other_eq_dist_focal_plus_other': (df['f_other'] == df['dist_focal'] + df['other']).mean(),
    'f_other_eq_focal_plus_f_focal': (df['f_other'] == df['focal'] + df['f_focal']).mean(),
    'win_eq_dist_focal_plus_other': (df['win'] == df['dist_focal'] + df['other']).mean(),
    'win_eq_focal_plus_f_focal': (df['win'] == df['focal'] + df['f_focal']).mean(),
}
print(checks)

# Also check if f_other equals dist_focal + other for all rows etc
for k, v in checks.items():
    print(k, v)

# find exact columns for IDs: n_other, dist_other presumably focal/other ids, dyad id
print(df[['n_other','dist_other','dyad']].nunique())

# compute differences for possible size mapping

