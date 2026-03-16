import pandas as pd
import itertools

_df = pd.read_csv('crofoot.csv')

male_cols = ['dist_focal','focal']
female_cols = ['other','f_focal']
total_cols = ['f_other','win']

for total in total_cols:
    best = (0,None)
    for m in male_cols:
        for f in female_cols:
            match = (_df[m] + _df[f] == _df[total]).mean()
            if match > best[0]:
                best = (match,(m,f))
    print(f'{total}: best match {best[0]:.2f} from {best[1]}')
