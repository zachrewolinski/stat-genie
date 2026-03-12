import json
import numpy as np
import pandas as pd
from scipy import stats


df = pd.read_csv('reading.csv')
for col in ['reader_view','speed','dyslexia_bin']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

dys = df[df['dyslexia_bin']==1].copy()
dys = dys.dropna(subset=['uuid','reader_view','speed'])

# Within-subject: mean speed by uuid and reader_view
pivot = dys.pivot_table(index='uuid', columns='reader_view', values='speed', aggfunc='mean')
# Keep subjects with both conditions 0 and 1
paired = pivot.dropna(subset=[0,1])

rv1 = paired[1]
rv0 = paired[0]

paired_summary = {
    'n_paired': int(paired.shape[0]),
    'mean_rv1': float(rv1.mean()) if paired.shape[0] else np.nan,
    'mean_rv0': float(rv0.mean()) if paired.shape[0] else np.nan,
    'mean_diff': float((rv1-rv0).mean()) if paired.shape[0] else np.nan,
    'median_diff': float((rv1-rv0).median()) if paired.shape[0] else np.nan
}

if paired.shape[0] > 1:
    ttest_rel = stats.ttest_rel(rv1, rv0, nan_policy='omit')
    ttest_rel_res = {'t_stat': float(ttest_rel.statistic), 'p_value': float(ttest_rel.pvalue)}
    try:
        wilcoxon = stats.wilcoxon(rv1, rv0)
        wilcoxon_res = {'stat': float(wilcoxon.statistic), 'p_value': float(wilcoxon.pvalue)}
    except Exception:
        wilcoxon_res = {'stat': np.nan, 'p_value': np.nan}
else:
    ttest_rel_res = {'t_stat': np.nan, 'p_value': np.nan}
    wilcoxon_res = {'stat': np.nan, 'p_value': np.nan}

results = {
    'paired_summary': paired_summary,
    'paired_ttest': ttest_rel_res,
    'paired_wilcoxon': wilcoxon_res,
    'n_total_dys': int(dys.shape[0]),
    'n_unique_dys': int(dys['uuid'].nunique())
}

print(json.dumps(results, indent=2))
