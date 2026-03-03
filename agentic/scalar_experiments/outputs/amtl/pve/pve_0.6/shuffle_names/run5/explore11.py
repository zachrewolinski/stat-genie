import pandas as pd
amtl = pd.read_csv('amtl.csv')
print('corr genus-age', amtl['genus'].corr(amtl['age']))
