import pandas as pd
import numpy as np

path = 'affairs.csv'
df = pd.read_csv(path)
# Map yes/no to 1/0
if df['religiousness'].dtype == object:
    df['religiousness_bin'] = (df['religiousness'] == 'yes').astype(int)

# compute correlations between continuous columns and binned versions
cont_cols = ['education', 'age']
other_cols = ['occupation', 'children', 'yearsmarried', 'rownames', 'rating', 'affairs']
for c in cont_cols:
    for o in other_cols:
        if df[o].dtype != object:
            corr = np.corrcoef(df[c], df[o])[0,1]
            print(f"corr({c},{o})={corr:.3f}")

# also check distribution of education and age
print('education describe', df['education'].describe())
print('age describe', df['age'].describe())

# Check unique values for rating and affairs counts
print('rating values', sorted(df['rating'].unique()))
print('affairs values', sorted(df['affairs'].unique()))

