import pandas as pd
import numpy as np

_df = pd.read_csv('amtl.csv')

sum_genus_round = _df.groupby('prob_male')['genus'].apply(lambda x: np.round(x).sum())
num_amtl_round = _df.groupby('prob_male')['num_amtl'].apply(lambda x: np.round(x.iloc[0]))

print('corr(sum(round(genus)), round(num_amtl))', sum_genus_round.corr(num_amtl_round))

# difference distribution
diff = sum_genus_round - num_amtl_round
print(diff.describe())
print('fraction within 1:', (diff.abs()<=1).mean())
print('fraction within 2:', (diff.abs()<=2).mean())
