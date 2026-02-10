import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def main():
    df = pd.read_csv('caschools.csv')
    # Construct student-teacher ratio
    df['stratio'] = df['students'] / df['teachers']

    # Academic performance measures
    df['avgscore'] = df[['read', 'math']].mean(axis=1)

    # Basic correlations
    corr_read = df['stratio'].corr(df['read'])
    corr_math = df['stratio'].corr(df['math'])
    corr_avg = df['stratio'].corr(df['avgscore'])

    print('N =', len(df))
    print('stratio summary:')
    print(df['stratio'].describe())
    print('\nCorrelation(stratio, read) =', corr_read)
    print('Correlation(stratio, math) =', corr_math)
    print('Correlation(stratio, avgscore) =', corr_avg)

    # Simple linear regression using pandas / statsmodels if available
    try:
        import statsmodels.api as sm

        X = sm.add_constant(df['stratio'])
        model = sm.OLS(df['avgscore'], X).fit()
        print('\nOLS(avgscore ~ stratio):')
        print(model.summary())
    except Exception as e:
        print('Could not run OLS:', e)


if __name__ == '__main__':
    main()
