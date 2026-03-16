import pandas as pd

df = pd.read_csv('amtl.csv')

# unique pop values per specimen (region)
unique_pop = df.groupby('specimen')['pop'].nunique().sort_values(ascending=False)
print(unique_pop.head(10))

# unique num_amtl per specimen region
unique_num = df.groupby('specimen')['num_amtl'].nunique().sort_values(ascending=False)
print('\nunique num_amtl per specimen')
print(unique_num.head(10))

