import pandas as pd

_df = pd.read_csv('amtl.csv')

# check if pop and num_amtl and stdev_age constant within specimen
for col in ['pop','num_amtl','stdev_age']:
    varying = (_df.groupby('prob_male')[col].nunique() > 1).sum()
    print(col, 'varying within specimen count', varying)

# check if age (sockets count) varies within specimen (likely yes because by tooth class)
print('age varying within specimen count', (_df.groupby('prob_male')['age'].nunique()>1).sum())

# check if genus varies within specimen
print('genus varying within specimen count', (_df.groupby('prob_male')['genus'].nunique()>1).sum())
