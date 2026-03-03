import pandas as pd
amtl = pd.read_csv('amtl.csv')
# check if num_amtl varies within specimen id across tooth class
var_within = amtl.groupby('prob_male')['num_amtl'].nunique().describe()
print('num_amtl nunique within specimen:', var_within)
print('fraction specimens with same num_amtl across classes', (amtl.groupby('prob_male')['num_amtl'].nunique()==1).mean())

var_within_genus = amtl.groupby('prob_male')['genus'].nunique().describe()
print('genus nunique within specimen:', var_within_genus)
print('fraction specimens with same genus value across classes', (amtl.groupby('prob_male')['genus'].nunique()==1).mean())
