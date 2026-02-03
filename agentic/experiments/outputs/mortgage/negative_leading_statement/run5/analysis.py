import pandas as pd
import numpy as np
import statsmodels.api as sm


def main():
    df = pd.read_csv('mortgage.csv')

    # Core features to control for applicant and loan characteristics
    features = [
        'female',
        'black',
        'housing_expense_ratio',
        'self_employed',
        'married',
        'mortgage_credit',
        'consumer_credit',
        'bad_history',
        'PI_ratio',
        'loan_to_value',
    ]

    # Simple acceptance rate comparison (unadjusted)
    simple_df = df[['female', 'accept']].dropna()
    accept_rates = simple_df.groupby('female')['accept'].mean()
    counts = simple_df['female'].value_counts().sort_index()

    print('Unadjusted acceptance rates by gender (female=1):')
    print(accept_rates)
    print('Counts:', counts.to_dict())
    print()

    # Adjusted model: denial probability
    model_df = df[['deny'] + features].dropna()
    X = sm.add_constant(model_df[features])
    y = model_df['deny']

    logit_model = sm.Logit(y, X).fit(disp=0)

    coef = logit_model.params['female']
    pval = logit_model.pvalues['female']
    odds_ratio = float(np.exp(coef))

    print('Adjusted logit model on denial (deny=1):')
    print(f"female coefficient: {coef:.6f}")
    print(f"female p-value: {pval:.6f}")
    print(f"female odds ratio: {odds_ratio:.4f}")


if __name__ == '__main__':
    main()
