import pandas as pd
import numpy as np
import statsmodels.api as sm
import patsy
from math import erf

def main():
    df = pd.read_csv('amtl.csv')

    # Ensure categorical types
    for col in ['genus', 'tooth_class']:
        df[col] = df[col].astype('category')

    # Set Homo sapiens as reference category for genus
    if 'Homo sapiens' in df['genus'].cat.categories:
        cats = ['Homo sapiens'] + [c for c in df['genus'].cat.categories if c != 'Homo sapiens']
        df['genus'] = df['genus'].cat.reorder_categories(cats, ordered=False)

    # Design matrix for predictors
    exog = patsy.dmatrix('genus + age + prob_male + tooth_class', data=df, return_type='dataframe')

    # Binomial outcomes: successes and failures per specimen/tooth_class row
    success = df['num_amtl']
    failure = df['sockets'] - df['num_amtl']
    endog = np.column_stack([success, failure])

    model = sm.GLM(endog, exog, family=sm.families.Binomial())
    res = model.fit()

    print(res.summary())

    # One-sided tests: non-human genus coefficients < 0 implies Homo higher AMTL
    params = res.params
    bse = res.bse
    print('\npairwise non-human vs Homo (coef negative => Homo higher):')
    for g in ['Pan', 'Pongo', 'Papio']:
        term = f'genus[T.{g}]'
        if term in params.index:
            z = params[term] / bse[term]
            p_one_sided = 0.5 * (1 + erf(z / np.sqrt(2)))
            print(f'{g}: coef={params[term]:.4f}, se={bse[term]:.4f}, z={z:.3f}, one-sided p={p_one_sided:.3g}')

    # Predicted AMTL probability at mean covariates and common tooth class
    mean_age = df['age'].mean()
    mean_male = df['prob_male'].mean()
    common_tc = df['tooth_class'].mode()[0]

    pred_rows = []
    for g in ['Homo sapiens', 'Pan', 'Pongo', 'Papio']:
        pred_rows.append({'genus': g, 'age': mean_age, 'prob_male': mean_male, 'tooth_class': common_tc})
    pred_df = pd.DataFrame(pred_rows)
    pred_df['genus'] = pred_df['genus'].astype('category').cat.set_categories(df['genus'].cat.categories)
    pred_df['tooth_class'] = pred_df['tooth_class'].astype('category').cat.set_categories(df['tooth_class'].cat.categories)

    pred_exog = patsy.dmatrix('genus + age + prob_male + tooth_class', data=pred_df, return_type='dataframe')
    linpred = pred_exog @ params
    pred_prob = 1 / (1 + np.exp(-linpred))

    print('\npredicted AMTL probability (mean covariates, common tooth class):')
    for g, p in zip(pred_df['genus'], pred_prob):
        print(f'{g}: {p:.4f}')

if __name__ == '__main__':
    main()
