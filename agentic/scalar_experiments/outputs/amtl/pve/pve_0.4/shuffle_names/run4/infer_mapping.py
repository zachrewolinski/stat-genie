import pandas as pd
import numpy as np

# Load data

df = pd.read_csv('amtl.csv')

# correlations
for col in ['age','pop','num_amtl','stdev_age']:
    print('corr genus with', col, df['genus'].corr(df[col]))

# test if genus is logit of num_amtl/age
ratio = df['genus']

# compute some candidate transforms
candidates = {
    'log_num_amtl': np.log(df['num_amtl']),
    'log_pop': np.log(df['pop']),
    'log_age': np.log(df['age']),
    'log_ratio_num_amtl_age': np.log(df['num_amtl'] / df['age']),
    'log_ratio_genus_age': np.log((df['genus']+1e-6) / df['age']),
    'logit_num_amtl_over_age': np.log((df['num_amtl'] / df['age']).clip(1e-6,1-1e-6) / (1-(df['num_amtl'] / df['age']).clip(1e-6,1-1e-6))),
    'num_amtl_over_age': df['num_amtl'] / df['age'],
    'genus_over_age': df['genus'] / df['age'],
}

for name, val in candidates.items():
    corr = df['genus'].corr(val)
    print(name, corr)

# check if genus is close to num_amtl/age (linear relationship)
from sklearn.linear_model import LinearRegression
import numpy as np

X = df[['num_amtl','age']]
model = LinearRegression().fit(X, df['genus'])
print('R2 genus ~ num_amtl + age', model.score(X, df['genus']))
print('coef', model.coef_, 'intercept', model.intercept_)

# check if num_amtl is close to genus*age
pred = df['genus'] * df['age']
print('corr num_amtl with genus*age', df['num_amtl'].corr(pred))

