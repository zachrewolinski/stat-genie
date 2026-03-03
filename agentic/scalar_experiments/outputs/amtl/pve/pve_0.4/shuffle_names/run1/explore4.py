import pandas as pd
import numpy as np

df = pd.read_csv('amtl.csv')

zpop = (df['pop'] - df['pop'].mean())/df['pop'].std()
print('corr(genus, zpop)=', df['genus'].corr(zpop))

znum = (df['num_amtl'] - df['num_amtl'].mean())/df['num_amtl'].std()
print('corr(genus, znum)=', df['genus'].corr(znum))

# check linear fit
coef = np.polyfit(zpop, df['genus'], 1)
print('genus ~ zpop coef', coef)

# check if genus approx zpop with offset
print('genus mean', df['genus'].mean(), 'std', df['genus'].std())

# compute max abs diff between genus and zpop*std+mean? not

# check if genus equals log(pop)
logpop = np.log(df['pop'])
print('corr(genus, log(pop))', df['genus'].corr(logpop))

# check if genus is log(num_amtl)
lognum = np.log(df['num_amtl'])
print('corr(genus, log(num_amtl))', df['genus'].corr(lognum))

