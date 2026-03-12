import pandas as pd
import statsmodels.formula.api as smf
import numpy as np

# Load data

df = pd.read_csv('amtl.csv')

# Ensure categorical
for col in ['genus','tooth_class','specimen']:
    df[col] = df[col].astype('category')

# OLS model with controls
formula = "num_amtl ~ C(genus, Treatment(reference='Homo sapiens')) + age + prob_male + C(tooth_class)"
model = smf.ols(formula, data=df)
res = model.fit(cov_type='cluster', cov_kwds={'groups': df['specimen']})
print(res.summary())

# Extract genus coefficients
params = res.params
pvalues = res.pvalues
for term in params.index:
    if term.startswith('C(genus'):
        print(term, 'coef', params[term], 'p', pvalues[term])

# Joint test that non-human genera coefficients are zero (i.e., no difference vs Homo)
# Equivalent to: C(genus)[T.Pan] = 0, C(genus)[T.Papio] = 0, C(genus)[T.Pongo] = 0
# Using F-test
hypotheses = []
for genus in ['Pan','Papio','Pongo']:
    term = f"C(genus, Treatment(reference='Homo sapiens'))[T.{genus}]"
    if term in params.index:
        hypotheses.append(term + ' = 0')

if hypotheses:
    ftest = res.f_test(hypotheses)
    print('Joint test for non-human genera vs Homo:')
    print(ftest)

# Compute predicted mean for each genus at mean age, mean prob_male, and marginal over tooth_class (average of classes)
mean_age = df['age'].mean()
mean_prob_male = df['prob_male'].mean()

tooth_classes = df['tooth_class'].cat.categories

def pred_for_genus(genus):
    preds = []
    for tc in tooth_classes:
        tmp = pd.DataFrame({
            'genus': [genus],
            'age': [mean_age],
            'prob_male': [mean_prob_male],
            'tooth_class': [tc]
        })
        preds.append(res.predict(tmp)[0])
    return np.mean(preds)

preds = {g: pred_for_genus(g) for g in df['genus'].cat.categories}
print('Predicted mean num_amtl by genus (at mean age/prob_male, avg tooth class):')
for g, v in preds.items():
    print(g, v)

# Differences vs Homo
homo = preds['Homo sapiens']
for g, v in preds.items():
    if g != 'Homo sapiens':
        print('Diff', g, 'vs Homo', v - homo)

