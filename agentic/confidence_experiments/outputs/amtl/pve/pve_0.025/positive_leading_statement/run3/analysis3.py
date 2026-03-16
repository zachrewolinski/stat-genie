import pandas as pd
import statsmodels.formula.api as smf

amtl = pd.read_csv('amtl.csv')

amtl['genus'] = amtl['genus'].astype('category')
amtl['tooth_class'] = amtl['tooth_class'].astype('category')
amtl['genus'] = amtl['genus'].cat.reorder_categories(['Homo sapiens','Pan','Pongo','Papio'], ordered=False)

model = smf.ols('num_amtl ~ C(genus) + age + prob_male + C(tooth_class)', data=amtl).fit(
    cov_type='cluster', cov_kwds={'groups': amtl['specimen']}
)
print(model.summary())
