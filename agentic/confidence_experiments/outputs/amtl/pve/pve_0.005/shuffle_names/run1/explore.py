import pandas as pd

_df = pd.read_csv('amtl.csv')
# count rows per specimen id (prob_male)
counts = _df['prob_male'].value_counts().describe()
print('rows per prob_male describe', counts)
# check distribution of age per specimen
print('age unique per prob_male', _df.groupby('prob_male')['age'].nunique().describe())
print('pop unique per prob_male', _df.groupby('prob_male')['pop'].nunique().describe())
print('num_amtl unique per prob_male', _df.groupby('prob_male')['num_amtl'].nunique().describe())
print('stdev_age unique per prob_male', _df.groupby('prob_male')['stdev_age'].nunique().describe())
# check tooth class per prob_male
print('sockets unique per prob_male', _df.groupby('prob_male')['sockets'].nunique().describe())

# check if num_amtl <= pop or age? for each row
print('num_amtl <= age proportion', (_df['num_amtl'] <= _df['age']).mean())
print('num_amtl <= pop proportion', (_df['num_amtl'] <= _df['pop']).mean())
print('genus <= age proportion', (_df['genus'] <= _df['age']).mean())
print('genus <= pop proportion', (_df['genus'] <= _df['pop']).mean())

# check if genus values near 0-1? no.

