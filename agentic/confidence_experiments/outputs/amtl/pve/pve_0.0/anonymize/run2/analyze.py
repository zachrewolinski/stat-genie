import pandas as pd
import statsmodels.formula.api as smf
import numpy as np

# Load data

df = pd.read_csv('amtl.csv')

# Rename for readability

df = df.rename(columns={
    'feature1': 'tooth_class',
    'feature3': 'amtl_z',
    'feature4': 'sockets',
    'feature5': 'age',
    'feature6': 'age_uncert',
    'feature7': 'sex',
    'feature8': 'genus',
})

# Ensure categorical types
for col in ['tooth_class', 'genus']:
    df[col] = df[col].astype('category')

# OLS model controlling for tooth class, age, sex, and sockets
# Use Homo sapiens as reference to directly compare others to humans.

model = smf.ols('amtl_z ~ C(genus, Treatment(reference="Homo sapiens")) + C(tooth_class) + age + sex + sockets', data=df).fit()

print(model.summary())

# Extract coefficients for genus comparisons
coef = model.params
pvals = model.pvalues

comparisons = {
    'Pan': 'C(genus, Treatment(reference="Homo sapiens"))[T.Pan]',
    'Pongo': 'C(genus, Treatment(reference="Homo sapiens"))[T.Pongo]',
    'Papio': 'C(genus, Treatment(reference="Homo sapiens"))[T.Papio]',
}

print('\nGenus comparisons vs Homo sapiens:')
for genus, term in comparisons.items():
    if term in coef:
        print(genus, coef[term], pvals[term])

# Estimated marginal means for each genus at average covariates and most common tooth_class
mean_vals = {
    'age': df['age'].mean(),
    'sex': df['sex'].mean(),
    'sockets': df['sockets'].mean(),
}

# Use overall distribution of tooth_class by averaging predictions across classes

classes = df['tooth_class'].cat.categories

preds = {}
for genus in df['genus'].cat.categories:
    pred_list = []
    for tc in classes:
        row = {
            'genus': genus,
            'tooth_class': tc,
            **mean_vals,
        }
        pred_list.append(model.predict(pd.DataFrame([row]))[0])
    preds[genus] = float(np.mean(pred_list))

print('\nPredicted mean amtl_z by genus (avg covariates):')
for genus, pred in preds.items():
    print(genus, pred)

# Pairwise difference Homo - others
homo = preds['Homo sapiens']
print('\nPredicted differences (Homo - other):')
for genus, pred in preds.items():
    if genus != 'Homo sapiens':
        print(genus, homo - pred)
