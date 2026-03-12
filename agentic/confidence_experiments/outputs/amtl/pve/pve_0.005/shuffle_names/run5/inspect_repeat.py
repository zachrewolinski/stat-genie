import pandas as pd

df = pd.read_csv('amtl.csv')
# number of unique num_amtl per specimen
uniq_counts = df.groupby('prob_male')['num_amtl'].nunique()
print(uniq_counts.value_counts().head())
# number of unique genus per specimen
print(df.groupby('prob_male')['genus'].nunique().value_counts().head())
# number of unique age per specimen
print(df.groupby('prob_male')['age'].nunique().value_counts().head())
