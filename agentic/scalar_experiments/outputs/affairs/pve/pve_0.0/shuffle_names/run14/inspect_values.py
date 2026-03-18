import pandas as pd

path='affairs.csv'
df=pd.read_csv(path)

# show unique values for potentially categorical numeric vars
for col in ['age','occupation','children','rating','yearsmarried','rownames','affairs','education']:
    vals=sorted(df[col].unique())
    print(col, len(vals), vals[:15], '...', vals[-15:])

# show value counts for object columns
for col in ['gender','religiousness']:
    print(col, df[col].value_counts())

