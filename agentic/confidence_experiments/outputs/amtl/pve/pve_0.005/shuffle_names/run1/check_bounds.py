import pandas as pd
_df = pd.read_csv('amtl.csv')
# proportion in [0, age]
prop = ((_df['genus'] >= 0) & (_df['genus'] <= _df['age'])).mean()
print('genus in [0, age] proportion', prop)
print('genus >=0 proportion', (_df['genus']>=0).mean())
# for num_amtl as missing maybe? check [0, age]
prop2 = ((_df['num_amtl'] >= 0) & (_df['num_amtl'] <= _df['age'])).mean()
print('num_amtl in [0, age] proportion', prop2)
print('num_amtl >=0 proportion', (_df['num_amtl']>=0).mean())

