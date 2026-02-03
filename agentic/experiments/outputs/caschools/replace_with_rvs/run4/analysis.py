import pandas as pd
import statsmodels.api as sm

def main():
    df = pd.read_csv('caschools.csv')
    # student-teacher ratio
    df['str'] = df['students'] / df['teachers']
    # academic performance: average of read and math
    df['score_avg'] = df[['read', 'math']].mean(axis=1)

    # correlation
    corr = df['str'].corr(df['score_avg'])

    # OLS regression: score_avg ~ str
    X = sm.add_constant(df['str'])
    model = sm.OLS(df['score_avg'], X).fit()

    print('corr_str_score_avg', corr)
    print(model.summary().as_text())

if __name__ == '__main__':
    main()
