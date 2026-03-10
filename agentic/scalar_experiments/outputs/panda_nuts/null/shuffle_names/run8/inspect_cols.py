import pandas as pd

df = pd.read_csv('panda_nuts.csv')

df = df.rename(columns={
    'help': 'nuts_opened_count',
    'chimpanzee': 'seconds',
    'nuts_opened': 'sex',
    'seconds': 'help_received',
})

print(df['sex'].head())
print(df['help_received'].head())

print('sex unique', df['sex'].unique())
print('help_received unique', df['help_received'].unique())
print('sex types', {type(x) for x in df['sex'].head(10)})
print('help types', {type(x) for x in df['help_received'].head(10)})
