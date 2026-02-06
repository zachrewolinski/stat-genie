import pandas as pd
import numpy as np
from statsmodels.stats.weightstats import ttest_ind
import statsmodels.formula.api as smf


def main():
    df = pd.read_csv('reading.csv')

    # Focus on participants with dyslexia
    sub = df[df['dyslexia_bin'] == 1].copy()
    sub = sub[(sub['speed'] > 0) & sub['reader_view'].notna()]
    sub['log_speed'] = np.log(sub['speed'])

    # Descriptive stats
    counts = sub['reader_view'].value_counts().sort_index()
    means = sub.groupby('reader_view')['speed'].mean()
    medians = sub.groupby('reader_view')['speed'].median()

    # T-test on log speed (more stable than raw speed)
    rv0 = sub[sub['reader_view'] == 0]['log_speed']
    rv1 = sub[sub['reader_view'] == 1]['log_speed']
    t_stat, p_val, dfree = ttest_ind(rv1, rv0, usevar='unequal')

    # Regression with content/device controls (robust SEs)
    cols = ['log_speed', 'reader_view', 'page_id', 'num_words', 'Flesch_Kincaid', 'device']
    sub2 = sub[cols].dropna()
    model = smf.ols(
        'log_speed ~ reader_view + C(page_id) + num_words + Flesch_Kincaid + C(device)',
        data=sub2,
    ).fit(cov_type='HC3')

    print('Dyslexia sample size:', len(sub))
    print('Reader view counts (0,1):', counts.to_dict())
    print('Mean speed by reader_view:', means.to_dict())
    print('Median speed by reader_view:', medians.to_dict())
    print('T-test log speed (t, p, df):', (t_stat, p_val, dfree))
    print('Regression reader_view coef:', model.params['reader_view'])
    print('Regression reader_view p-value:', model.pvalues['reader_view'])


if __name__ == '__main__':
    main()
