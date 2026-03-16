import pandas as pd

DATA_PATH = 'panda_nuts.csv'

df = pd.read_csv(DATA_PATH)
print(df.head())
print('\nunique nuts_opened:', df['nuts_opened'].unique())
print('unique sex:', df['sex'].unique())
print('unique seconds:', df['seconds'].unique())
print('min/max age', df['age'].min(), df['age'].max())
print('min/max hammer', df['hammer'].min(), df['hammer'].max())
print('min/max help', df['help'].min(), df['help'].max())
print('min/max chimpanzee', df['chimpanzee'].min(), df['chimpanzee'].max())
print('count rows', len(df))
