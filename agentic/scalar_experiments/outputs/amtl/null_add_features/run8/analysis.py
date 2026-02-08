import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main():
    df = pd.read_csv('amtl.csv')

    # Keep relevant columns
    cols = ['num_amtl', 'sockets', 'age', 'prob_male', 'tooth_class', 'genus']
    df = df[cols].copy()

    # Basic cleaning
    df = df.dropna()
    df = df[(df['sockets'] > 0) & (df['num_amtl'] >= 0)]
    df = df[df['num_amtl'] <= df['sockets']]

    df['human'] = (df['genus'] == 'Homo sapiens').astype(int)

    # Binomial response as successes/failures
    df['failures'] = df['sockets'] - df['num_amtl']

    # Fit GLM
    formula = 'num_amtl + failures ~ human + age + prob_male + C(tooth_class)'
    model = smf.glm(formula=formula, data=df, family=sm.families.Binomial()).fit()

    # Extract human effect
    human_coef = model.params.get('human', np.nan)
    human_p = model.pvalues.get('human', np.nan)

    # Average marginal effect on probability using counterfactual predictions
    base = df.copy()
    base['human'] = 0
    pred_non = model.predict(base)
    base['human'] = 1
    pred_human = model.predict(base)
    ame = (pred_human - pred_non).mean()

    # Report
    print('n_rows', len(df))
    print('human_coef_logit', human_coef)
    print('human_pvalue', human_p)
    print('ame_prob', ame)


if __name__ == '__main__':
    main()
