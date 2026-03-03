import pandas as pd
amtl = pd.read_csv('amtl.csv')
print(amtl.groupby('tooth_class')['stdev_age'].value_counts())
