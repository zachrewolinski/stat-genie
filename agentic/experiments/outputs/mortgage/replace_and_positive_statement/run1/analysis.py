import pandas as pd
import numpy as np
import statsmodels.api as sm


def logistic_regression(df, y_col, x_cols):
    X = df[x_cols].copy()
    X = sm.add_constant(X, has_constant='add')
    y = df[y_col]
    model = sm.Logit(y, X)
    result = model.fit(disp=False)
    return result


def main():
    df = pd.read_csv('mortgage.csv')

    # Basic checks
    print('Rows:', len(df))
    print('Columns:', list(df.columns))
    print('\nFemale summary:')
    print(df['female'].describe())

    # Simple association: correlation with accept
    corr = df['female'].corr(df['accept'])
    print('\nCorrelation (female, accept):', corr)

    # Unadjusted logistic regression
    unadj = logistic_regression(df, 'accept', ['female'])
    print('\nUnadjusted logistic regression: accept ~ female')
    print(unadj.summary())

    # Adjusted logistic regression with common underwriting controls
    controls = [
        'female', 'black', 'housing_expense_ratio', 'self_employed', 'married',
        'mortgage_credit', 'consumer_credit', 'bad_history', 'PI_ratio',
        'loan_to_value', 'denied_PMI'
    ]
    adj = logistic_regression(df, 'accept', controls)
    print('\nAdjusted logistic regression: accept ~ female + controls')
    print(adj.summary())

    # Effect size: change in predicted probability for +1 std in female
    female_std = df['female'].std()
    X_mean = df[controls].mean().to_frame().T
    X_mean = sm.add_constant(X_mean, has_constant='add')
    base_prob = adj.predict(X_mean).iloc[0]
    X_shift = X_mean.copy()
    X_shift['female'] = X_shift['female'] + female_std
    shift_prob = adj.predict(X_shift).iloc[0]
    print('\nPredicted approval probability at mean controls:', base_prob)
    print('Predicted approval probability with female +1 std:', shift_prob)
    print('Difference (percentage points):', (shift_prob - base_prob) * 100)


if __name__ == '__main__':
    main()
