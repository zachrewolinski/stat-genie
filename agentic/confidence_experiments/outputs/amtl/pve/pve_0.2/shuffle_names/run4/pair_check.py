import pandas as pd
import itertools

cols=['genus','age','pop','num_amtl','stdev_age']
df=pd.read_csv('amtl.csv')

for miss in cols:
    for sock in cols:
        if miss==sock: continue
        m=df[miss]
        s=df[sock]
        # require miss>=0, s>=0, miss<=s
        ok=((m>=0)&(s>=0)&(m<=s)).mean()
        print(f"{miss} as missing <= {sock}: {ok:.3f}")
