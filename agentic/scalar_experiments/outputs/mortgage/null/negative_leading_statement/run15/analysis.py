import json
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats


def main():
    df = pd.read_csv('mortgage.csv')
    # Basic cleaning
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])

    # Ensure expected columns
    # Outcome: deny (1 denied, 0 accepted)
    # Explanatory: female + controls
    # Drop rows with missing values in relevant columns
    cols = [
        'deny', 'female', 'black', 'housing_expense_ratio', 'self_employed',
        'married', 'mortgage_credit', 'consumer_credit', 'bad_history',
        'PI_ratio', 'loan_to_value', 'denied_PMI'
    ]
    missing_cols = [c for c in cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns: {missing_cols}")
    dfa = df[cols].dropna()

    # Unadjusted denial rates by gender
    denial_by_gender = dfa.groupby('female')['deny'].agg(['mean', 'count'])

    # Logistic regression with controls
    formula = (
        'deny ~ female + black + housing_expense_ratio + self_employed + married + '
        'mortgage_credit + consumer_credit + bad_history + PI_ratio + loan_to_value + denied_PMI'
    )
    model = smf.logit(formula=formula, data=dfa).fit(disp=False, cov_type='HC1')

    # Extract female coefficient and stats (robust from cov_type)
    coef = model.params['female']
    se = model.bse['female']
    z = coef / se
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    odds_ratio = float(np.exp(coef))
    ci_low = float(np.exp(coef - 1.96 * se))
    ci_high = float(np.exp(coef + 1.96 * se))

    # Also compute average marginal effect of female
    try:
        marg = model.get_margeff(at='overall')
        marg_df = marg.summary_frame()
        ame = float(marg_df.loc['female', 'dy/dx'])
        ame_se = float(marg_df.loc['female', 'Std. Err.'])
        ame_z = float(marg_df.loc['female', 'z'])
        ame_p = float(marg_df.loc['female', 'Pr(>|z|)'])
    except Exception:
        ame = ame_se = ame_z = ame_p = np.nan

    results = {
        'n': int(dfa.shape[0]),
        'denial_rate_female': float(denial_by_gender.loc[1.0, 'mean']) if 1.0 in denial_by_gender.index else None,
        'denial_rate_male': float(denial_by_gender.loc[0.0, 'mean']) if 0.0 in denial_by_gender.index else None,
        'count_female': int(denial_by_gender.loc[1.0, 'count']) if 1.0 in denial_by_gender.index else None,
        'count_male': int(denial_by_gender.loc[0.0, 'count']) if 0.0 in denial_by_gender.index else None,
        'logit_female_coef': float(coef),
        'logit_female_se': float(se),
        'logit_female_z': float(z),
        'logit_female_p': float(p),
        'logit_female_odds_ratio': odds_ratio,
        'logit_female_or_ci_low': ci_low,
        'logit_female_or_ci_high': ci_high,
        'ame_female': float(ame) if not np.isnan(ame) else None,
        'ame_female_se': float(ame_se) if not np.isnan(ame_se) else None,
        'ame_female_z': float(ame_z) if not np.isnan(ame_z) else None,
        'ame_female_p': float(ame_p) if not np.isnan(ame_p) else None,
    }

    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
