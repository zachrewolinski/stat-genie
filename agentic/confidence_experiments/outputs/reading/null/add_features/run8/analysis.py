import json
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm

DATA_PATH = 'reading.csv'


def cohen_d(x, y):
    x = np.asarray(x)
    y = np.asarray(y)
    nx = x.size
    ny = y.size
    if nx < 2 or ny < 2:
        return np.nan
    vx = x.var(ddof=1)
    vy = y.var(ddof=1)
    pooled = ((nx - 1) * vx + (ny - 1) * vy) / (nx + ny - 2)
    if pooled <= 0:
        return np.nan
    return (x.mean() - y.mean()) / np.sqrt(pooled)


def main():
    df = pd.read_csv(DATA_PATH)

    # Focus on dyslexic participants
    if 'dyslexia_bin' in df.columns:
        dys_df = df[df['dyslexia_bin'] == 1].copy()
    else:
        dys_df = df[df['dyslexia'] > 0].copy()

    # Ensure speed and reader_view are numeric
    dys_df = dys_df[pd.to_numeric(dys_df['speed'], errors='coerce').notna()]
    dys_df = dys_df[pd.to_numeric(dys_df['reader_view'], errors='coerce').notna()]
    dys_df['speed'] = dys_df['speed'].astype(float)
    dys_df['reader_view'] = dys_df['reader_view'].astype(int)

    # Remove extreme outliers using 1st/99th percentile to reduce undue influence
    if len(dys_df) > 0:
        lo, hi = dys_df['speed'].quantile([0.01, 0.99])
        dys_df = dys_df[(dys_df['speed'] >= lo) & (dys_df['speed'] <= hi)]

    # Group statistics
    speed_rv = dys_df[dys_df['reader_view'] == 1]['speed']
    speed_no = dys_df[dys_df['reader_view'] == 0]['speed']

    group_stats = {
        'n_reader_view': int(speed_rv.shape[0]),
        'n_no_view': int(speed_no.shape[0]),
        'mean_reader_view': float(speed_rv.mean()) if speed_rv.shape[0] else np.nan,
        'mean_no_view': float(speed_no.mean()) if speed_no.shape[0] else np.nan,
        'median_reader_view': float(speed_rv.median()) if speed_rv.shape[0] else np.nan,
        'median_no_view': float(speed_no.median()) if speed_no.shape[0] else np.nan,
    }

    # Independent t-test and Mann-Whitney U
    test_results = {}
    if speed_rv.shape[0] >= 2 and speed_no.shape[0] >= 2:
        t_res = stats.ttest_ind(speed_rv, speed_no, equal_var=False, nan_policy='omit')
        mw_res = stats.mannwhitneyu(speed_rv, speed_no, alternative='two-sided')
        test_results['t_stat'] = float(t_res.statistic)
        test_results['t_pvalue'] = float(t_res.pvalue)
        test_results['mw_u'] = float(mw_res.statistic)
        test_results['mw_pvalue'] = float(mw_res.pvalue)
        test_results['cohen_d'] = float(cohen_d(speed_rv, speed_no))
    else:
        test_results['t_stat'] = np.nan
        test_results['t_pvalue'] = np.nan
        test_results['mw_u'] = np.nan
        test_results['mw_pvalue'] = np.nan
        test_results['cohen_d'] = np.nan

    # Within-subject comparison if participants have both conditions
    paired_results = {}
    if 'uuid' in dys_df.columns:
        pivot = dys_df.pivot_table(index='uuid', columns='reader_view', values='speed', aggfunc='mean')
        if 0 in pivot.columns and 1 in pivot.columns:
            paired = pivot.dropna()
            if paired.shape[0] >= 5:
                diff = paired[1] - paired[0]
                t_paired = stats.ttest_rel(paired[1], paired[0])
                paired_results = {
                    'n_paired': int(paired.shape[0]),
                    'mean_diff': float(diff.mean()),
                    'median_diff': float(diff.median()),
                    't_stat': float(t_paired.statistic),
                    't_pvalue': float(t_paired.pvalue),
                }
            else:
                paired_results = {'n_paired': int(paired.shape[0])}

    # Simple regression controlling for num_words (if available)
    reg_results = {}
    if 'num_words' in dys_df.columns and dys_df['num_words'].notna().sum() > 0:
        reg_df = dys_df[['speed', 'reader_view', 'num_words']].dropna().copy()
        if reg_df.shape[0] >= 30:
            # log-transform speed to reduce skew
            reg_df['log_speed'] = np.log(reg_df['speed'].clip(lower=1))
            X = sm.add_constant(reg_df[['reader_view', 'num_words']])
            model = sm.OLS(reg_df['log_speed'], X).fit()
            reg_results = {
                'n': int(reg_df.shape[0]),
                'coef_reader_view': float(model.params['reader_view']),
                'p_reader_view': float(model.pvalues['reader_view']),
                'r2': float(model.rsquared),
            }

    results = {
        'group_stats': group_stats,
        'test_results': test_results,
        'paired_results': paired_results,
        'reg_results': reg_results,
    }

    with open('analysis_results.json', 'w') as f:
        json.dump(results, f, indent=2)


if __name__ == '__main__':
    main()
