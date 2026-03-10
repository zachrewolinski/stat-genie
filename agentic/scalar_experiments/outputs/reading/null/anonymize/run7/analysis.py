import pandas as pd
import numpy as np
import pingouin as pg


def compute_reading_speed(df):
    # words per minute using reading time minus scrolling (feature5) in ms
    speed = df['feature7'] / (df['feature5'] / 60000.0)
    return speed


def select_cols(df, cols):
    return df[[c for c in cols if c in df.columns]]


def paired_analysis(df, dyslexia_filter, label):
    sub = df[dyslexia_filter].copy()
    # aggregate per participant and condition
    agg = (
        sub.groupby(['feature1', 'feature3'])['reading_speed_wpm']
        .median()
        .reset_index()
    )
    # pivot to wide
    wide = agg.pivot(index='feature1', columns='feature3', values='reading_speed_wpm')
    # require both conditions 0 and 1
    wide = wide.dropna(subset=[0, 1])
    # paired t-test
    if wide.shape[0] >= 3:
        ttest = pg.ttest(wide[1], wide[0], paired=True)
        # wilcoxon for robustness
        try:
            wilcoxon = pg.wilcoxon(wide[1], wide[0])
        except Exception:
            wilcoxon = None
    else:
        ttest = None
        wilcoxon = None
    # effect size (paired Cohen's d)
    diff = wide[1] - wide[0]
    mean_diff = diff.mean() if len(diff) else np.nan
    return {
        'label': label,
        'n_participants': wide.shape[0],
        'mean_speed_no_reader': wide[0].mean() if wide.shape[0] else np.nan,
        'mean_speed_reader': wide[1].mean() if wide.shape[0] else np.nan,
        'median_speed_no_reader': wide[0].median() if wide.shape[0] else np.nan,
        'median_speed_reader': wide[1].median() if wide.shape[0] else np.nan,
        'mean_diff_reader_minus_no': mean_diff,
        'ttest': ttest,
        'wilcoxon': wilcoxon,
        'diff_series': diff,
    }


def paired_analysis_f20(df, dyslexia_filter, label):
    sub = df[dyslexia_filter].copy()
    agg = (
        sub.groupby(['feature1', 'feature3'])['reading_speed_f20']
        .median()
        .reset_index()
    )
    wide = agg.pivot(index='feature1', columns='feature3', values='reading_speed_f20')
    wide = wide.dropna(subset=[0, 1])
    if wide.shape[0] >= 3:
        ttest = pg.ttest(wide[1], wide[0], paired=True)
        try:
            wilcoxon = pg.wilcoxon(wide[1], wide[0])
        except Exception:
            wilcoxon = None
    else:
        ttest = None
        wilcoxon = None
    diff = wide[1] - wide[0]
    return {
        'label': label,
        'n_participants': wide.shape[0],
        'mean_speed_no_reader': wide[0].mean() if wide.shape[0] else np.nan,
        'mean_speed_reader': wide[1].mean() if wide.shape[0] else np.nan,
        'median_speed_no_reader': wide[0].median() if wide.shape[0] else np.nan,
        'median_speed_reader': wide[1].median() if wide.shape[0] else np.nan,
        'mean_diff_reader_minus_no': diff.mean() if len(diff) else np.nan,
        'ttest': ttest,
        'wilcoxon': wilcoxon,
    }


def print_test(label, res):
    print('\n', label)
    print('n=', res['n_participants'])
    print('mean no reader', res['mean_speed_no_reader'])
    print('mean reader', res['mean_speed_reader'])
    print('median no reader', res['median_speed_no_reader'])
    print('median reader', res['median_speed_reader'])
    print('mean diff reader-no', res['mean_diff_reader_minus_no'])
    if res['ttest'] is not None:
        print('ttest', select_cols(res['ttest'], ['T','dof','p-val','cohen-d','CI95%']))
    if res['wilcoxon'] is not None:
        print('wilcoxon', select_cols(res['wilcoxon'], ['W-val','p-val','RBC','CI95%']))


def main():
    df = pd.read_csv('reading.csv')

    # basic cleaning
    df = df.copy()
    df['reading_speed_wpm'] = compute_reading_speed(df)
    # filter valid rows
    df = df[
        (df['feature5'] > 0) &
        (df['feature7'] > 0) &
        (df['feature3'].isin([0, 1])) &
        (df['reading_speed_wpm'].replace([np.inf, -np.inf], np.nan).notna())
    ].copy()

    # define dyslexia groups
    dys_bin = df['feature17'] == 1
    dys_severe = df['feature12'] == 2
    dys_any = df['feature12'].isin([1, 2])

    results = []
    results.append(paired_analysis(df, dys_bin, 'feature17==1 (dyslexia)'))
    results.append(paired_analysis(df, dys_any, 'feature12 in {1,2} (dyslexia or severe)'))
    results.append(paired_analysis(df, dys_severe, 'feature12==2 (severe dyslexia)'))

    # Also run sensitivity with feature20 as reading speed if available
    df2 = df.copy()
    df2['reading_speed_f20'] = df2['feature20']

    results_f20 = []
    results_f20.append(paired_analysis_f20(df2, dys_bin, 'feature17==1 (dyslexia)'))
    results_f20.append(paired_analysis_f20(df2, dys_any, 'feature12 in {1,2} (dyslexia or severe)'))
    results_f20.append(paired_analysis_f20(df2, dys_severe, 'feature12==2 (severe dyslexia)'))

    # Print results
    print('=== Derived reading speed (WPM from feature5 and feature7) ===')
    for res in results:
        print_test(res['label'], res)

    print('\n=== Feature20 as reading speed (sensitivity) ===')
    for res in results_f20:
        print_test(res['label'], res)


if __name__ == '__main__':
    main()
