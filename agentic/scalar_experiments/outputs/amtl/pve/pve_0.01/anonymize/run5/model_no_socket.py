import pandas as pd
import statsmodels.formula.api as smf

_df = pd.read_csv('amtl.csv')
_df['is_human'] = (_df['feature8'] == 'Homo sapiens').astype(int)

model1 = smf.ols('feature3 ~ is_human + feature5 + feature7 + C(feature1)', data=_df).fit()
print(model1.summary())
print('is_human coef', model1.params['is_human'], 'p', model1.pvalues['is_human'])

model2 = smf.ols('feature3 ~ C(feature8, Treatment(reference="Homo sapiens")) + feature5 + feature7 + C(feature1)', data=_df).fit()
print(model2.summary())
for term in model2.params.index:
    if 'C(feature8' in term:
        print(term, 'coef', model2.params[term], 'p', model2.pvalues[term])
