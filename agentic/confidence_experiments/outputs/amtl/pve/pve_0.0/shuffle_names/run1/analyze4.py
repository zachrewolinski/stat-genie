import pandas as pd

pd.set_option('display.max_rows', 20)

df = pd.read_csv('amtl.csv')

specimen = df['prob_male'].iloc[0]
print('specimen', specimen)
print(df[df['prob_male'] == specimen])

# check if num_amtl varies by sockets within specimen
var_by_specimen = df.groupby('prob_male')[['genus','age','pop','num_amtl','stdev_age']].nunique()
print('\nWithin-specimen unique counts (first 5):')
print(var_by_specimen.head())
print('\nProportion of specimens where num_amtl varies across sockets:', (var_by_specimen['num_amtl']>1).mean())
print('Proportion where age varies:', (var_by_specimen['age']>1).mean())
print('Proportion where pop varies:', (var_by_specimen['pop']>1).mean())
print('Proportion where genus varies:', (var_by_specimen['genus']>1).mean())

