import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from statsmodels.stats.weightstats import ttest_ind


def main():
    df = pd.read_csv('reading.csv')

    # Focus on participants with dyslexia
    df = df[df['dyslexia_bin'] == 1].copy()

    # Basic cleaning
    df = df[df['speed'].notna() & df['reader_view'].notna()]
    df = df[df['speed'] > 0]

    # Group summaries
    summary = df.groupby('reader_view')['speed'].agg(['count', 'mean', 'median', 'std'])
    print('Speed summary for dyslexic participants by reader_view (0=off, 1=on):')
    print(summary)
    print()

    # Welch t-test on raw speed
    g0 = df[df['reader_view'] == 0]['speed']
    g1 = df[df['reader_view'] == 1]['speed']
    t_stat, p_val, _ = ttest_ind(g1, g0, usevar='unequal')
    print('Welch t-test on speed (reader_view=1 vs 0):')
    print(f't={t_stat:.3f}, p={p_val:.4g}, mean_diff={g1.mean() - g0.mean():.3f}')
    print()

    # Log-speed to reduce skew
    df['log_speed'] = np.log(df['speed'])
    g0_log = df[df['reader_view'] == 0]['log_speed']
    g1_log = df[df['reader_view'] == 1]['log_speed']
    t_stat_log, p_val_log, _ = ttest_ind(g1_log, g0_log, usevar='unequal')
    print('Welch t-test on log(speed) (reader_view=1 vs 0):')
    print(f't={t_stat_log:.3f}, p={p_val_log:.4g}, mean_log_diff={g1_log.mean() - g0_log.mean():.4f}')
    print()

    # Regression with controls and participant clustering
    # Keep rows with non-missing covariates used below
    model_df = df[[
        'log_speed', 'reader_view', 'page_id', 'device', 'age', 'gender',
        'education', 'english_native', 'retake_trial', 'correct_rate', 'uuid'
    ]].dropna().copy()

    formula = (
        'log_speed ~ reader_view + C(page_id) + C(device) + age + C(gender) '
        '+ C(education) + C(english_native) + retake_trial + correct_rate'
    )
    model = smf.ols(formula, data=model_df).fit(
        cov_type='cluster', cov_kwds={'groups': model_df['uuid']}
    )

    coef = model.params['reader_view']
    se = model.bse['reader_view']
    p = model.pvalues['reader_view']
    pct = (np.exp(coef) - 1) * 100

    print('Regression on log(speed) with controls (cluster-robust by participant):')
    print(f'reader_view coef={coef:.4f}, SE={se:.4f}, p={p:.4g}, approx % change={pct:.2f}%')
    print(f'n={model_df.shape[0]} observations, {model_df["uuid"].nunique()} participants')


if __name__ == '__main__':
    main()
