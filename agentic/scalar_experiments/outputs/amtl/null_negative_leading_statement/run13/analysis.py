import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
import patsy

# Load data
_df = pd.read_csv('amtl.csv')

# Basic cleaning
_df = _df.copy()
_df = _df[_df['sockets'] > 0]
_df = _df[_df['num_amtl'] <= _df['sockets']]

# Design matrix
formula = "C(genus) + age + prob_male + C(tooth_class)"
exog = patsy.dmatrix(formula, _df, return_type='dataframe')
endog = np.column_stack([_df['num_amtl'].values, (_df['sockets'] - _df['num_amtl']).values])

model = sm.GLM(endog, exog, family=sm.families.Binomial())
res = model.fit()

# Marginal standardization to get adjusted AMTL frequency per genus

def adjusted_rate_for_genus(genus_value, df=_df, res=res, design_info=exog.design_info):
    df2 = df.copy()
    df2['genus'] = genus_value
    exog2 = patsy.build_design_matrices([design_info], df2, return_type='dataframe')[0]
    p = res.predict(exog2)
    # weight by sockets to represent tooth-level frequency
    w = df2['sockets'].values
    return np.sum(p * w) / np.sum(w)

# Compute adjusted rates
unique_genera = sorted(_df['genus'].unique())
rate_by_genus = {g: adjusted_rate_for_genus(g) for g in unique_genera}

# Non-human genera list
non_human = [g for g in unique_genera if g != 'Homo sapiens']

# Simple mean of adjusted rates across non-human genera
nonhuman_mean = float(np.mean([rate_by_genus[g] for g in non_human]))

# Differences
homo_rate = rate_by_genus.get('Homo sapiens', np.nan)

# Bootstrap for uncertainty
rng = np.random.default_rng(7)
B = 300
boot_diffs = []
boot_diffs_vs = {g: [] for g in non_human}

rows = _df.shape[0]
for _ in range(B):
    idx = rng.integers(0, rows, rows)
    dfb = _df.iloc[idx].copy()
    exog_b = patsy.dmatrix(formula, dfb, return_type='dataframe')
    endog_b = np.column_stack([dfb['num_amtl'].values, (dfb['sockets'] - dfb['num_amtl']).values])
    try:
        res_b = sm.GLM(endog_b, exog_b, family=sm.families.Binomial()).fit()
    except Exception:
        # rare convergence issue; skip
        continue

    def adj_rate(genus_value):
        df2 = dfb.copy()
        df2['genus'] = genus_value
        exog2 = patsy.build_design_matrices([exog_b.design_info], df2, return_type='dataframe')[0]
        p = res_b.predict(exog2)
        w = df2['sockets'].values
        return np.sum(p * w) / np.sum(w)

    rate_h = adj_rate('Homo sapiens') if 'Homo sapiens' in dfb['genus'].unique() else np.nan
    rates_non = [adj_rate(g) for g in non_human]
    if np.any(np.isnan(rates_non)) or np.isnan(rate_h):
        continue
    diff = rate_h - float(np.mean(rates_non))
    boot_diffs.append(diff)
    for g in non_human:
        boot_diffs_vs[g].append(rate_h - adj_rate(g))

boot_diffs = np.array(boot_diffs)

# Summaries
summary = {
    'adjusted_rates': rate_by_genus,
    'homo_minus_nonhuman_mean': float(homo_rate - nonhuman_mean),
    'bootstrap_n': int(boot_diffs.size),
}
if boot_diffs.size > 0:
    summary['diff_ci_95'] = [float(np.quantile(boot_diffs, 0.025)), float(np.quantile(boot_diffs, 0.975))]

for g in non_human:
    arr = np.array(boot_diffs_vs[g])
    if arr.size > 0:
        summary[f'homo_minus_{g}_ci_95'] = [float(np.quantile(arr, 0.025)), float(np.quantile(arr, 0.975))]

with open('analysis_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

print(json.dumps(summary, indent=2))
