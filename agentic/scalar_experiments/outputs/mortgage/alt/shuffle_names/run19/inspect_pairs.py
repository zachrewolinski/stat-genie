import pandas as pd, json

df = pd.read_csv('mortgage.csv')

pairs = [('deny','self_employed'), ('deny','accept'), ('self_employed','accept')]
res = {}
for a,b in pairs:
    if a in df.columns and b in df.columns:
        s = df[a] + df[b]
        res[f'{a}+{b}'] = {
            'mean_sum': float(s.mean()),
            'prop_sum_1': float((s==1).mean()),
            'prop_sum_0': float((s==0).mean()),
            'prop_sum_2': float((s==2).mean()),
            'corr': float(df[a].corr(df[b])),
        }

print(json.dumps(res, indent=2))
