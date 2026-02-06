import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main():
    df = pd.read_csv('panda_nuts.csv')

    # Efficiency: nuts opened per second
    df['efficiency'] = df['nuts_opened'] / df['seconds']

    # Ensure categorical variables are treated as categories
    df['sex'] = df['sex'].astype('category')
    df['help'] = df['help'].astype('category')

    # OLS regression: efficiency ~ age + sex + help
    model = smf.ols('efficiency ~ age + C(sex) + C(help)', data=df).fit(cov_type='HC3')

    print(model.summary())

    # Extract p-values for predictors of interest
    pvals = model.pvalues
    print("\nP-values:")
    print(pvals)

    # Robustness check: Poisson model for counts with offset log(seconds)
    df['log_seconds'] = np.log(df['seconds'])
    poisson_model = smf.glm(
        'nuts_opened ~ age + C(sex) + C(help)',
        data=df,
        family=sm.families.Poisson(),
        offset=df['log_seconds']
    ).fit(cov_type='HC3')

    print("\nPoisson model summary (nuts_opened with log(seconds) offset):")
    print(poisson_model.summary())


if __name__ == '__main__':
    main()
