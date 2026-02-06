import pandas as pd
import statsmodels.formula.api as smf


def main():
    df = pd.read_csv('teachingratings.csv')

    # Simple model: ratings vs beauty
    m1 = smf.ols('allstudents ~ beauty', data=df).fit()

    # Controls: treat string columns as categorical
    # Exclude likely identifiers (division, students)
    formula = (
        'allstudents ~ beauty + age + rownames + minority '
        '+ C(eval) + C(tenure) + C(prof) + C(native) + C(gender) + C(credits)'
    )
    m2 = smf.ols(formula, data=df).fit()

    print('N:', len(df))
    print('\nSimple model: allstudents ~ beauty')
    print(m1.summary().tables[1])
    print('\nControlled model: allstudents ~ beauty + controls')
    print(m2.summary().tables[1])

    # Extra: effect size for 1 SD increase in beauty
    beauty_sd = df['beauty'].std()
    coef_simple = m1.params['beauty']
    coef_ctrl = m2.params['beauty']
    print('\nBeauty SD:', beauty_sd)
    print('Simple model: 1 SD beauty effect on rating:', coef_simple * beauty_sd)
    print('Controlled model: 1 SD beauty effect on rating:', coef_ctrl * beauty_sd)


if __name__ == '__main__':
    main()
