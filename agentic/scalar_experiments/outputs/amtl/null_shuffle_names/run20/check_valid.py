import pandas as pd

raw = pd.read_csv("amtl.csv")

df = raw.rename(columns={
    "sockets": "tooth_class",
    "prob_male": "specimen_id",
    "genus": "num_amtl",
    "age": "num_sockets",
    "pop": "age_at_death",
    "num_amtl": "stdev_age",
    "stdev_age": "prob_male",
    "tooth_class": "genus",
    "specimen": "region",
})

print("num_sockets min", df['num_sockets'].min())
print("num_sockets zeros", (df['num_sockets']==0).sum())
print("num_sockets negatives", (df['num_sockets']<0).sum())
print("num_amtl zeros", (df['num_amtl']==0).sum())
print("num_amtl negatives", (df['num_amtl']<0).sum())

# invalid rows for binomial: num_sockets<=0 or num_amtl>num_sockets
invalid = (df['num_sockets']<=0) | (df['num_amtl']>df['num_sockets'])
print("invalid rows", invalid.sum())
print(df.loc[invalid, ['num_amtl','num_sockets']].head())
