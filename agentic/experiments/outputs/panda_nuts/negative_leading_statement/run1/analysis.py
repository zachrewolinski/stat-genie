import pandas as pd
import statsmodels.formula.api as smf
import numpy as np


def main():
    df = pd.read_csv('panda_nuts.csv')

    # Efficiency: nuts opened per second
    df['efficiency'] = df['nuts_opened'] / df['seconds']

    # Basic OLS with categorical predictors; cluster-robust SE by chimpanzee
    model = smf.ols('efficiency ~ age + C(sex) + C(help)', data=df)
    result = model.fit(cov_type='cluster', cov_kwds={'groups': df['chimpanzee']})

    # Also fit on log(1+efficiency) to check robustness to skew
    df['log_efficiency'] = np.log1p(df['efficiency'])
    model_log = smf.ols('log_efficiency ~ age + C(sex) + C(help)', data=df)
    result_log = model_log.fit(cov_type='cluster', cov_kwds={'groups': df['chimpanzee']})

    print('OLS efficiency (cluster-robust SE by chimpanzee):')
    print(result.summary())
    print('\nOLS log(1+efficiency) (cluster-robust SE by chimpanzee):')
    print(result_log.summary())

    # Save key results for reference
    key = pd.DataFrame({
        'term': result.params.index,
        'coef_efficiency': result.params.values,
        'p_efficiency': result.pvalues.values,
        'coef_log_efficiency': result_log.params.values,
        'p_log_efficiency': result_log.pvalues.values,
    })
    key.to_csv('analysis_results.csv', index=False)


if __name__ == '__main__':
    main()
