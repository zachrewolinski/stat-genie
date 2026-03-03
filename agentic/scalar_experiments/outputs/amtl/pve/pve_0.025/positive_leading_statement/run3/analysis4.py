import pandas as pd
import statsmodels.formula.api as smf

amtl = pd.read_csv('amtl.csv')

amtl['human'] = (amtl['genus'] == 'Homo sapiens').astype(int)
amtl['tooth_class'] = amtl['tooth_class'].astype('category')

model = smf.ols('num_amtl ~ human + age + prob_male + C(tooth_class)', data=amtl).fit(
    cov_type='cluster', cov_kwds={'groups': amtl['specimen']}
)
print(model.summary())

# simple adjusted difference with human vs non-human; compute mean difference unadjusted too
print('Unadjusted mean num_amtl human', amtl.loc[amtl['human']==1,'num_amtl'].mean())
print('Unadjusted mean num_amtl non-human', amtl.loc[amtl['human']==0,'num_amtl'].mean())

