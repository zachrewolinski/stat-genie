import pandas as pd

_df = pd.read_csv('amtl.csv')

age = _df['age']

for col in ['genus','pop','num_amtl','stdev_age']:
    # check proportion of rows where value <= age
    le = ( _df[col] <= age ).mean()
    le_round = ( _df[col].round() <= age ).mean()
    le_floor = ( _df[col].apply(lambda x:int(x)) <= age ).mean()
    print(col, 'frac <= age', le, 'frac round<=age', le_round, 'frac int<=age', le_floor)
