import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from scipy.stats import chi2


def lrt(full, reduced):
    lr_stat = 2 * (full.llf - reduced.llf)
    df = full.df_model - reduced.df_model
    p = chi2.sf(lr_stat, df)
    return lr_stat, df, p


def main():
    df = pd.read_csv('boxes.csv')

    # Define outcomes
    df['social_choice'] = df['y'].isin([2, 3]).astype(int)  # chose demonstrated option
    df['majority_choice'] = (df['y'] == 2).astype(int)      # chose majority option

    # Descriptives
    overall = {
        'n': len(df),
        'social_choice_rate': df['social_choice'].mean(),
        'majority_choice_rate': df['majority_choice'].mean(),
        'majority_among_social_rate': df.loc[df['social_choice'] == 1, 'majority_choice'].mean(),
    }

    # Age groups for descriptive summaries
    bins = [3, 6, 9, 12, 14]
    labels = ['4-6', '7-9', '10-12', '13-14']
    df['age_group'] = pd.cut(df['age'], bins=bins, labels=labels)

    desc_age = df.groupby('age_group').agg(
        n=('y', 'size'),
        social_choice_rate=('social_choice', 'mean'),
        majority_choice_rate=('majority_choice', 'mean'),
    )
    desc_culture = df.groupby('culture').agg(
        n=('y', 'size'),
        social_choice_rate=('social_choice', 'mean'),
        majority_choice_rate=('majority_choice', 'mean'),
        majority_among_social_rate=('majority_choice', lambda s: s[df.loc[s.index, 'social_choice'] == 1].mean()),
    )

    # Modeling: reliance on social info
    # Reduced model: age + culture
    # Full model: age * culture
    m_social_reduced = smf.logit('social_choice ~ age + C(culture)', data=df).fit(disp=0)
    m_social_full = smf.logit('social_choice ~ age * C(culture)', data=df).fit(disp=0)
    lrt_social = lrt(m_social_full, m_social_reduced)

    # Modeling: majority preference among those who chose a demonstrated option
    df_social = df[df['social_choice'] == 1].copy()
    m_major_reduced = smf.logit('majority_choice ~ age + C(culture)', data=df_social).fit(disp=0)
    m_major_full = smf.logit('majority_choice ~ age * C(culture)', data=df_social).fit(disp=0)
    lrt_major = lrt(m_major_full, m_major_reduced)

    # Also test main effects (age, culture) in reduced models
    # Use likelihood ratio by dropping each term from reduced model
    m_social_age_only = smf.logit('social_choice ~ age', data=df).fit(disp=0)
    m_social_cult_only = smf.logit('social_choice ~ C(culture)', data=df).fit(disp=0)
    lrt_social_culture = lrt(m_social_reduced, m_social_age_only)
    lrt_social_age = lrt(m_social_reduced, m_social_cult_only)

    m_major_age_only = smf.logit('majority_choice ~ age', data=df_social).fit(disp=0)
    m_major_cult_only = smf.logit('majority_choice ~ C(culture)', data=df_social).fit(disp=0)
    lrt_major_culture = lrt(m_major_reduced, m_major_age_only)
    lrt_major_age = lrt(m_major_reduced, m_major_cult_only)

    print('Overall rates:', overall)
    print('\nBy age group:\n', desc_age)
    print('\nBy culture:\n', desc_culture)

    print('\nSocial reliance model (social_choice ~ age + culture)')
    print(m_social_reduced.summary())
    print('LRT age effect within age+culture model:', lrt_social_age)
    print('LRT culture effect within age+culture model:', lrt_social_culture)
    print('LRT interaction (age*culture):', lrt_social)

    print('\nMajority preference among social choices (majority_choice ~ age + culture)')
    print(m_major_reduced.summary())
    print('LRT age effect within age+culture model:', lrt_major_age)
    print('LRT culture effect within age+culture model:', lrt_major_culture)
    print('LRT interaction (age*culture):', lrt_major)


if __name__ == '__main__':
    main()
