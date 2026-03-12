import pandas as pd
amtl = pd.read_csv('amtl.csv')
print('corr genus-num_amtl', amtl['genus'].corr(amtl['num_amtl']))
print('corr genus-pop', amtl['genus'].corr(amtl['pop']))
print('corr num_amtl-pop', amtl['num_amtl'].corr(amtl['pop']))
