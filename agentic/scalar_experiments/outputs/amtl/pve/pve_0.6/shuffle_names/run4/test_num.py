import pandas as pd
import statsmodels.formula.api as smf

raw = pd.read_csv('amtl.csv')
df = raw.rename(columns={'genus':'amtl','pop':'age_at_death','stdev_age':'prob_male','sockets':'tooth_class','tooth_class':'genus'})

for col in ['age_at_death','prob_male']:
    try:
        smf.ols(f'amtl ~ {col}', data=df).fit()
        print('OK', col, df[col].dtype)
    except Exception as e:
        print('FAIL', col, type(e), e, 'dtype', df[col].dtype)

