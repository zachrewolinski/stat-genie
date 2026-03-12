import pandas as pd
import statsmodels.formula.api as smf

# Load data
df = pd.read_csv("amtl.csv")

# Fit OLS with Homo sapiens as reference
formula = 'num_amtl ~ C(genus, Treatment(reference="Homo sapiens")) + age + prob_male + C(tooth_class)'
model = smf.ols(formula, data=df).fit(cov_type='HC3')

print(model.summary())

# Extract genus coefficients and p-values
params = model.params
pvalues = model.pvalues

results = []
for genus in ['Pan', 'Papio', 'Pongo']:
    term = f'C(genus, Treatment(reference="Homo sapiens"))[T.{genus}]'
    if term in params:
        results.append((genus, params[term], pvalues[term]))

print('\nGenus differences vs Homo sapiens (negative => Homo higher):')
for genus, coef, pval in results:
    print(f'{genus}: coef={coef:.3f}, p={pval:.4g}')

# Joint test: all three genus effects = 0
terms = [f'C(genus, Treatment(reference="Homo sapiens"))[T.{g}]' for g in ['Pan','Papio','Pongo']]
print('\nJoint F-test for genus differences vs Homo sapiens:')
print(model.f_test(" + ".join(terms) + ' = 0'))
