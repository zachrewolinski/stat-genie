import pandas as pd

df = pd.read_csv('affairs.csv')
# correlation between age column and affairs (marriage rating)
print('corr age vs affairs', df['age'].corr(df['affairs']))
print('corr education vs affairs', df['education'].corr(df['affairs']))
# compare age with religiousness (actually children) and rating (religiousness)
print('corr age vs religiousness (children yes/no)', df['age'].corr(df['religiousness'].map({'yes':1,'no':0})))
print('corr age vs rating (religiousness scale)', df['age'].corr(df['rating']))
