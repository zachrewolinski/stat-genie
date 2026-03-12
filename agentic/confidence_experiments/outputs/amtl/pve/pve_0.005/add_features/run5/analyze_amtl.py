import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from scipy import stats

_df = pd.read_csv('amtl.csv')
_df['genus'] = _df['genus'].astype('category')
_df['tooth_class'] = _df['tooth_class'].astype('category')


def fit_and_report(formula, label):
    model = smf.ols(formula, data=_df).fit(cov_type='HC3')
    print('\n===', label, '===')
    print(model.summary())

    coef = model.params
    pvals = model.pvalues

    print('\nGenus coefficients vs Homo sapiens:')
    for term in coef.index:
        if 'C(genus' in term:
            print(term, 'coef', coef[term], 'p', pvals[term])

    results = []
    for term in coef.index:
        if 'C(genus' in term:
            est = coef[term]
            se = model.bse[term]
            zstat = est / se
            p_one_sided = stats.norm.cdf(zstat)
            results.append((term, est, se, zstat, p_one_sided))

    print('\nOne-sided tests (other - Homo < 0):')
    for term, est, se, z, p1 in results:
        print(term, 'est', est, 'se', se, 'z', z, 'p_one_sided', p1)

    mean_age = _df['age'].mean()
    mean_prob_male = _df['prob_male'].mean()

    preds = {}
    for genus in _df['genus'].cat.categories:
        temp = _df[['age', 'prob_male', 'tooth_class']].copy()
        if 'sockets' in formula:
            temp['sockets'] = _df['sockets'].mean()
        temp['age'] = mean_age
        temp['prob_male'] = mean_prob_male
        temp['genus'] = genus
        preds[genus] = model.predict(temp).mean()

    print('\nPredicted mean num_amtl (averaged across tooth_class distribution):')
    for k, v in preds.items():
        print(k, v)

    homo = preds['Homo sapiens']
    for genus, val in preds.items():
        if genus != 'Homo sapiens':
            print('Homo -', genus, ':', homo - val)


# Base model per research question
formula_base = 'num_amtl ~ C(genus, Treatment(reference="Homo sapiens")) + age + prob_male + C(tooth_class)'
fit_and_report(formula_base, 'Base model')

# Sensitivity: add sockets as additional control
formula_sockets = 'num_amtl ~ C(genus, Treatment(reference="Homo sapiens")) + age + prob_male + C(tooth_class) + sockets'
fit_and_report(formula_sockets, 'Sensitivity model (+sockets)')

