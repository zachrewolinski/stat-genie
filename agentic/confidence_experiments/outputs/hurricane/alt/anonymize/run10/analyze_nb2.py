import pandas as pd
import statsmodels.formula.api as smf


df = pd.read_csv('hurricane.csv')
cols = {f'feature{i}': f'f{i}' for i in range(1, 15)}
df = df.rename(columns=cols)

for var in ['f4','f6','f12']:
    formula = f'f8 ~ {var} + f7 + f5 + f13'
    try:
        model = smf.negativebinomial(formula, data=df).fit(disp=False)
        print('\nDiscrete NegativeBinomial:', formula)
        print(model.summary())
    except Exception as e:
        print('Failed for', var, e)

