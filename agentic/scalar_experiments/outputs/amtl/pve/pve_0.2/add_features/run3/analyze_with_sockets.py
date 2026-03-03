import pandas as pd
import statsmodels.formula.api as smf

df = pd.read_csv('amtl.csv')
for col in ['genus','tooth_class']:
    df[col] = df[col].astype('category')
if 'Homo sapiens' in df['genus'].cat.categories:
    new_cats = ['Homo sapiens'] + [c for c in df['genus'].cat.categories if c != 'Homo sapiens']
    df['genus'] = df['genus'].cat.reorder_categories(new_cats, ordered=False)

formula = 'num_amtl ~ C(genus) + age + prob_male + C(tooth_class) + sockets'
model = smf.ols(formula, data=df).fit(cov_type='HC3')
print(model.summary())
print(model.params[[k for k in model.params.index if k.startswith('C(genus)')]])
print(model.pvalues[[k for k in model.params.index if k.startswith('C(genus)')]])
