import pandas as pd

path = 'soccer.csv'
df = pd.read_csv(path)

id_candidates = ['photoID','goals','club']  # short name, full name, photo id
for pid in id_candidates:
    tmp = df[[pid,'rater1','nExp']].dropna()
    # number of players where rater1 varies
    var_r1 = (tmp.groupby(pid)['rater1'].nunique() > 1).mean()
    var_r2 = (tmp.groupby(pid)['nExp'].nunique() > 1).mean()
    print(pid, 'prop varying rater1', var_r1, 'prop varying nExp', var_r2, 'num unique', tmp[pid].nunique())

