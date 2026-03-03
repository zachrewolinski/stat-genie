import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm

# Load data
_df = pd.read_csv('amtl.csv')

# Recode
_df['is_human'] = (_df['feature8'] == 'Homo sapiens').astype(int)

# Model 1: OLS with human indicator, controlling for age, sex, tooth class, and sockets
model1 = smf.ols('feature3 ~ is_human + feature5 + feature7 + C(feature1) + feature4', data=_df).fit()
print(model1.summary())

# Model 2: OLS with genus categorical (Homo sapiens baseline)
# Use treatment coding with Homo sapiens as reference
_df['feature8'] = _df['feature8'].astype('category')
model2 = smf.ols('feature3 ~ C(feature8, Treatment(reference="Homo sapiens")) + feature5 + feature7 + C(feature1) + feature4', data=_df).fit()
print(model2.summary())

# Extract human indicator p-value, coef
print('Model1 is_human coef', model1.params['is_human'], 'p', model1.pvalues['is_human'])

# Extract genus coefficients vs Homo sapiens
for term in model2.params.index:
    if 'C(feature8' in term:
        print(term, 'coef', model2.params[term], 'p', model2.pvalues[term])

