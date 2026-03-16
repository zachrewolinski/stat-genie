import pandas as pd
import itertools

_df = pd.read_csv('crofoot.csv')
cols = ['f_other','win','dist_focal','focal','other','f_focal']

for target in cols:
    best = (0,None)
    for a,b in itertools.combinations([c for c in cols if c!=target],2):
        match = (_df[a]+_df[b] == _df[target]).mean()
        if match > best[0]:
            best = (match,(a,b))
    print(f'{target}: best match {best[0]:.2f} from {best[1]}')
