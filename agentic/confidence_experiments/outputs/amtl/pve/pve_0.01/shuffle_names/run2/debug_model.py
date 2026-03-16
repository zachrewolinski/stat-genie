import pandas as pd
import statsmodels.formula.api as smf

DF = pd.read_csv('amtl.csv')
DF = DF.rename(columns={
    'genus': 'amtl_count',
    'age': 'sockets_count',
    'pop': 'age_at_death',
    'stdev_age': 'prob_male',
    'tooth_class': 'genus_cat',
    'sockets': 'tooth_class'
})

print(DF.dtypes)
print(DF.head())

# check if any object column has list/array
for col in DF.columns:
    if DF[col].dtype == 'object':
        print(col, 'example type', type(DF[col].iloc[0]))

# Try creating design matrix
formula = 'amtl_rate ~ C(genus_cat) + age_at_death + prob_male + C(tooth_class)'
DF['amtl_rate'] = DF['amtl_count'] / DF['sockets_count']

try:
    m = smf.wls(formula, data=DF, weights=DF['sockets_count']).fit()
    print('model fit ok')
except Exception as e:
    print('error', e)

