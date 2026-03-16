import pandas as pd

df = pd.read_csv('mortgage.csv')
for col in ['denied_PMI','female','consumer_credit']:
    if col in df.columns:
        print(col, df[col].value_counts(dropna=False).to_dict())
