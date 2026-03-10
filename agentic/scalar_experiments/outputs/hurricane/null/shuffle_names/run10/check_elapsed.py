import pandas as pd

df = pd.read_csv('hurricane.csv')
# check if masfem corresponds to 2013-year or 2015-year etc
for ref in [2013, 2014, 2015, 2012]:
    diff = (ref - df['wind']) - df['masfem']
    print(ref, diff.abs().mean(), diff.abs().max())
