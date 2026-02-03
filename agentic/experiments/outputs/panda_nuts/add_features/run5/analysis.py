import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main():
    df = pd.read_csv('panda_nuts.csv')
    df['rate'] = df['nuts_opened'] / df['seconds']

    # Poisson regression with exposure (seconds) to model nut-cracking rate
    poisson_model = smf.glm(
        'nuts_opened ~ age + C(sex) + C(help)',
        data=df,
        family=sm.families.Poisson(),
        offset=np.log(df['seconds'])
    ).fit()

    # Linear regression on rate for a simple interpretability check
    ols_model = smf.ols('rate ~ age + C(sex) + C(help)', data=df).fit()

    # Save key results for quick inspection if needed
    with open('analysis_results.txt', 'w', encoding='utf-8') as f:
        f.write('Poisson GLM (nuts_opened with offset log(seconds))\n')
        f.write(poisson_model.summary().as_text())
        f.write('\n\nOLS on rate (nuts_opened/seconds)\n')
        f.write(ols_model.summary().as_text())
        f.write('\n\nGroup mean rate by sex:\n')
        f.write(df.groupby('sex')['rate'].mean().to_string())
        f.write('\n\nGroup mean rate by help:\n')
        f.write(df.groupby('help')['rate'].mean().to_string())


if __name__ == '__main__':
    main()
