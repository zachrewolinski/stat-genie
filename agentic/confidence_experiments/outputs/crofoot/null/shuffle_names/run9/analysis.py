import pandas as pd
import numpy as np
import statsmodels.api as sm


def main():
    df = pd.read_csv('crofoot.csv')

    # Map variables based on metadata descriptions (names were shuffled)
    y = df['m_focal']  # 1 if focal won contest

    focal_size = df['f_other']  # number of individuals in focal group
    other_size = df['win']      # number of individuals in other group
    rel_size = focal_size - other_size

    dist_focal = df['m_other']  # distance of focal group from center of its home range
    dist_other = df['n_focal']  # distance of other group from center of its home range
    rel_location = dist_focal - dist_other

    # Prepare design matrix
    X = pd.DataFrame({
        'rel_size': rel_size,
        'rel_location': rel_location,
    })
    X = sm.add_constant(X)

    model = sm.Logit(y, X)
    result = model.fit(disp=False)

    # Also fit model with standardized predictors for effect size interpretation
    Xz = (X[['rel_size', 'rel_location']] - X[['rel_size', 'rel_location']].mean()) / X[['rel_size', 'rel_location']].std(ddof=0)
    Xz = sm.add_constant(Xz)
    model_z = sm.Logit(y, Xz)
    result_z = model_z.fit(disp=False)

    # Compute marginal effect at mean for standardized predictors
    try:
        margeff = result_z.get_margeff(at='mean')
        me_table = margeff.summary_frame()
    except Exception:
        me_table = None

    print('N', len(df))
    print('\nLogit (raw predictors)')
    print(result.summary())
    print('\nLogit (standardized predictors)')
    print(result_z.summary())
    if me_table is not None:
        print('\nMarginal effects at mean (standardized predictors)')
        print(me_table)


if __name__ == '__main__':
    main()
