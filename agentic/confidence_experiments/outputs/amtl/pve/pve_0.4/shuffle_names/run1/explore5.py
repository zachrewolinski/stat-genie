import pandas as pd

df = pd.read_csv('amtl.csv')

# check variability within specimen id
spec = df.groupby('prob_male').agg({
    'pop':'nunique',
    'num_amtl':'nunique',
    'genus':'nunique',
    'age':'nunique',
    'stdev_age':'nunique',
    'tooth_class':'nunique',
    'sockets':'nunique'
})
print(spec.describe())
print('\nExamples with more than 1 unique within specimen for each variable:')
for col in ['pop','num_amtl','genus','age','stdev_age']:
    count = (spec[col] > 1).sum()
    print(col, 'specimens with >1 unique:', count)

# show typical for one specimen
sample = df[df['prob_male']==df['prob_male'].iloc[0]]
print('\nSample rows for specimen', df['prob_male'].iloc[0])
print(sample)
