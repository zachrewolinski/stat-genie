import pandas as pd
import numpy as np


df=pd.read_csv('amtl.csv')
# compute sum of age across classes per specimen
sum_age = df.groupby('prob_male')['age'].sum()
# check correlation with pop
pop = df.groupby('prob_male')['pop'].first()
print('corr sum(age) with pop', sum_age.corr(pop))
print('summary of sum_age', sum_age.describe())
print('summary of pop', pop.describe())
# check if pop equals sum_age maybe scaled
ratio = pop / sum_age
print('ratio pop/sum_age', ratio.describe())
