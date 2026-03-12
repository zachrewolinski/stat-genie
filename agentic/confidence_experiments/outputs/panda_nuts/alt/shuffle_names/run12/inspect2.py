import pandas as pd

pd.set_option('display.width', 200)

path = 'panda_nuts.csv'
df = pd.read_csv(path)
print('rows', len(df))
print('age unique', df['age'].nunique())
print('hammer unique', df['hammer'].nunique())

# check if age or hammer seems like ID by counting duplicates and group size
print('age counts top')
print(df['age'].value_counts().head())
print('hammer counts top')
print(df['hammer'].value_counts().head())

# check relationship: age -> hammer? maybe each age has single hammer? etc.
print('age->hammer unique counts')
print(df.groupby('age')['hammer'].nunique().describe())
print('hammer->age unique counts')
print(df.groupby('hammer')['age'].nunique().describe())

# show a couple groups
print('age group sample')
print(df.groupby('age')['hammer'].unique().head())
print('hammer group sample')
print(df.groupby('hammer')['age'].unique().head())
