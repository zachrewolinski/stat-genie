import pandas as pd
import numpy as np

df = pd.read_csv('affairs.csv')

# check if education or age are linear transforms of row index
idx = pd.Series(range(len(df)))
for col in ['education','age']:
    corr = df[col].corr(idx)
    print(col, 'corr with row index', corr)

# check basic stats and if values close to integer
for col in ['education','age']:
    vals = df[col].values
    frac = np.mean(np.isclose(vals, np.round(vals)))
    print(col, 'fraction near integer', frac)
    # percentiles
    print(col, 'percentiles', np.percentile(vals, [1,5,25,50,75,95,99]))

# check correlation between education and age
print('corr education-age', df['education'].corr(df['age']))

