import pandas as pd
import numpy as np
import statsmodels.api as sm
import patsy


def main():
    df = pd.read_csv('amtl.csv')

    # Rename for clarity
    df = df.rename(columns={
        'feature1': 'tooth_class',
        'feature2': 'specimen_id',
        'feature3': 'missing_teeth',
        'feature4': 'observable_sockets',
        'feature5': 'age',
        'feature6': 'age_uncertainty',
        'feature7': 'sex_estimate',
        'feature8': 'genus',
        'feature9': 'region'
    })

    # Clean and prepare
    df = df.dropna(subset=['missing_teeth', 'observable_sockets', 'age', 'sex_estimate', 'tooth_class', 'genus'])
    df = df[(df['observable_sockets'] > 0) & (df['missing_teeth'] >= 0)]
    df = df[df['missing_teeth'] <= df['observable_sockets']]

    df['present_teeth'] = df['observable_sockets'] - df['missing_teeth']
    df['is_human'] = (df['genus'] == 'Homo sapiens').astype(int)

    # Build design matrix
    formula = 'is_human + age + sex_estimate + C(tooth_class)'
    X = patsy.dmatrix(formula, df, return_type='dataframe')

    endog = np.column_stack([df['missing_teeth'], df['present_teeth']])

    model = sm.GLM(endog, X, family=sm.families.Binomial())
    result = model.fit()

    # Marginal effect of human indicator (difference in predicted probability)
    df_mean = df.copy()
    df_mean['is_human'] = 0
    y0 = result.predict(patsy.dmatrix(formula, df_mean, return_type='dataframe'))
    df_mean['is_human'] = 1
    y1 = result.predict(patsy.dmatrix(formula, df_mean, return_type='dataframe'))
    avg_diff = float(np.mean(y1 - y0))

    coef = result.params['is_human']
    pval = result.pvalues['is_human']
    odds_ratio = float(np.exp(coef))

    print(result.summary())
    print('\nKey effect: is_human coefficient')
    print(f'coef={coef:.4f}, odds_ratio={odds_ratio:.4f}, pval={pval:.4g}')
    print(f'Average marginal difference in missing probability: {avg_diff:.4f}')


if __name__ == '__main__':
    main()
