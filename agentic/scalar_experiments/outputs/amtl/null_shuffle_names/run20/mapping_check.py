import pandas as pd

raw = pd.read_csv("amtl.csv")

# Try mapping A: num_amtl=genus, num_sockets=age
num_amtl = raw['genus']
num_sockets = raw['age']
print('Mapping A exceed count', (num_amtl > num_sockets).sum())
print('Mapping A max excess', (num_amtl - num_sockets).max())

# mapping B: num_amtl=age, num_sockets=genus
num_amtl2 = raw['age']
num_sockets2 = raw['genus']
print('Mapping B exceed count', (num_amtl2 > num_sockets2).sum())
print('Mapping B max excess', (num_amtl2 - num_sockets2).max())

# mapping C: num_sockets maybe pop? (not integer) so unlikely

# list some rows where num_amtl > num_sockets for mapping A
excess = raw.loc[num_amtl > num_sockets, ['genus','age','tooth_class','sockets']].head(10)
print(excess)

# check min age for high genus
print(raw.groupby('genus')['age'].min().head(15))

# check if age values maybe represent age category and sockets count elsewhere
print(raw[['genus','age','pop','num_amtl','stdev_age']].head())
