import pandas as pd
amtl = pd.read_csv('amtl.csv')
print('num_amtl <= age fraction', (amtl['num_amtl'] <= amtl['age']).mean())
print('num_amtl max vs age max', amtl['num_amtl'].max(), amtl['age'].max())
print('num_amtl <= pop fraction', (amtl['num_amtl'] <= amtl['pop']).mean())
print('genus <= age fraction', (amtl['genus'] <= amtl['age']).mean())
print('genus <= pop fraction', (amtl['genus'] <= amtl['pop']).mean())
