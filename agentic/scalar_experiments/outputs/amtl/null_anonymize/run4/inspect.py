import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

col_map = {
    'feature1': 'tooth_class',
    'feature2': 'specimen_id',
    'feature3': 'missing',
    'feature4': 'observable',
    'feature5': 'age',
    'feature6': 'age_uncertainty',
    'feature7': 'sex',
    'feature8': 'genus',
    'feature9': 'region',
}

df = pd.read_csv('amtl.csv').rename(columns=col_map)
needed = ['missing', 'observable', 'age', 'sex', 'tooth_class', 'genus']
clean = df.dropna(subset=needed).copy()
clean['missing'] = pd.to_numeric(clean['missing'], errors='coerce')
clean['observable'] = pd.to_numeric(clean['observable'], errors='coerce')
clean['age'] = pd.to_numeric(clean['age'], errors='coerce')
clean['sex'] = pd.to_numeric(clean['sex'], errors='coerce')
clean = clean.dropna(subset=['missing','observable','age','sex'])
clean = clean[(clean['observable'] > 0) & (clean['missing'] >= 0) & (clean['missing'] <= clean['observable'])]
clean['present'] = clean['observable'] - clean['missing']
clean['genus'] = clean['genus'].astype('category')
clean['tooth_class'] = clean['tooth_class'].astype('category')

model = smf.glm('missing + present ~ C(genus) + age + sex + C(tooth_class)', data=clean, family=sm.families.Binomial()).fit()
print(model.params)
print(model.params.index)
print(clean['genus'].cat.categories)
