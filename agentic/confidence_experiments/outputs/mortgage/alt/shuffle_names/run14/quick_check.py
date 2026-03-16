import pandas as pd

df = pd.read_csv('mortgage.csv')
print(df.head())
print('columns', df.columns.tolist())
for col in ['female','accept','deny','denied_PMI','consumer_credit','bad_history','loan_to_value','married','black','self_employed','PI_ratio']:
    if col in df.columns:
        print('\n', col)
        print(df[col].describe())
        print(df[col].value_counts(dropna=False).head())
