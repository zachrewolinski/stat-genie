import pandas as pd
import statsmodels.formula.api as smf

amtl = pd.read_csv('amtl.csv')

cols = ['num_amtl', 'age', 'prob_male', 'tooth_class', 'genus']

df = amtl[cols].dropna().copy()

# Set Homo sapiens as reference category
# statsmodels uses alphabetical order by default; we'll set categorical with Homo sapiens first as reference

df['genus'] = pd.Categorical(df['genus'], categories=['Homo sapiens', 'Pan', 'Papio', 'Pongo'])

model = smf.ols('num_amtl ~ C(genus) + age + prob_male + C(tooth_class)', data=df).fit(cov_type='HC3')
print(model.summary())

# Extract coefficients for non-human vs Homo
for genus in ['Pan', 'Papio', 'Pongo']:
    term = f'C(genus)[T.{genus}]'
    coef = model.params[term]
    p = model.pvalues[term]
    print(genus, 'coef', coef, 'p', p)
