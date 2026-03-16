import pandas as pd
import numpy as np
import statsmodels.api as sm

# Load data
_df = pd.read_csv('crofoot.csv')

# Identify likely group-size columns by checking constancy within focal/other group IDs
focal_id = 'n_other'  # per metadata, this appears to be focal group ID (1-6)
other_id = 'dist_other'  # other group ID (1-6)

# Candidate group size columns: those with small set {5,6,10,13}
size_candidates = [c for c in _df.columns if set(_df[c].unique()) == {5, 6, 10, 13}]

# Check constancy within group IDs
constancy = {}
for col in size_candidates:
    focal_unique = _df.groupby(focal_id)[col].nunique().max()
    other_unique = _df.groupby(other_id)[col].nunique().max()
    constancy[col] = {'focal_unique_max': focal_unique, 'other_unique_max': other_unique}

print('Size candidates:', size_candidates)
print('Constancy within group IDs:', constancy)

# Choose mapping based on constancy: focal size should be constant within focal_id; other size within other_id
# We'll pick the column with focal_unique_max==1 as focal size; other_unique_max==1 as other size
focal_size_col = None
other_size_col = None
for col, stats in constancy.items():
    if stats['focal_unique_max'] == 1:
        focal_size_col = col
    if stats['other_unique_max'] == 1:
        other_size_col = col

print('Selected focal_size_col:', focal_size_col)
print('Selected other_size_col:', other_size_col)

# Distance columns (contest location) are those with large numeric ranges (>50)
dist_cols = [c for c in _df.columns if _df[c].max() > 50]
print('Distance columns:', dist_cols)

# By metadata, m_other is distance of focal group from center; n_focal is distance of other group from center
# Use those column names (since values match range) for location
# Determine which dist col corresponds to focal/other by constancy with IDs: distance from focal should vary, not constant, but may correlate? We'll follow metadata mapping
if set(dist_cols) == {'m_other', 'n_focal'}:
    dist_focal_col = 'm_other'
    dist_other_col = 'n_focal'
else:
    # fallback: choose two largest-range columns, order arbitrary
    dist_focal_col, dist_other_col = dist_cols[:2]

print('Selected dist_focal_col:', dist_focal_col)
print('Selected dist_other_col:', dist_other_col)

# Outcome
outcome_col = 'm_focal'  # binary

# Build analysis dataframe
_df = _df.copy()
_df['relative_size'] = _df[focal_size_col] - _df[other_size_col]
_df['relative_size_log'] = np.log(_df[focal_size_col] / _df[other_size_col])
_df['relative_location'] = _df[dist_focal_col] - _df[dist_other_col]

# Logistic regression: win ~ relative_size + relative_location
X = _df[['relative_size', 'relative_location']]
X = sm.add_constant(X)
model = sm.Logit(_df[outcome_col], X)
result = model.fit(disp=False)
print('\nLogit (relative_size, relative_location):')
print(result.summary())

# Alternative using log ratio
X2 = _df[['relative_size_log', 'relative_location']]
X2 = sm.add_constant(X2)
model2 = sm.Logit(_df[outcome_col], X2)
result2 = model2.fit(disp=False)
print('\nLogit (relative_size_log, relative_location):')
print(result2.summary())

# For effect sizes, compute predicted probability at +/-1 SD for each variable holding other at mean
for name, res, cols in [
    ('diff', result, ['relative_size', 'relative_location']),
    ('log', result2, ['relative_size_log', 'relative_location'])
]:
    means = _df[cols].mean()
    sds = _df[cols].std()
    base = means.copy()
    def pred(v):
        x = {'const':1.0}
        x.update(v)
        return res.predict(pd.DataFrame([x]))[0]
    for col in cols:
        v_low = base.copy(); v_low[col] = base[col] - sds[col]
        v_high = base.copy(); v_high[col] = base[col] + sds[col]
        p_low = pred(v_low)
        p_high = pred(v_high)
        print(f"Model {name}: {col} +/-1SD: {p_low:.3f} -> {p_high:.3f}")

# Simple comparisons: win rates by relative_size sign and relative_location sign
_df['rel_size_sign'] = np.sign(_df['relative_size'])
_df['rel_loc_sign'] = np.sign(_df['relative_location'])

print('\nWin rates by relative_size sign:')
print(_df.groupby('rel_size_sign')[outcome_col].mean())
print('\nWin rates by relative_location sign:')
print(_df.groupby('rel_loc_sign')[outcome_col].mean())

# Cross-tab
print('\nWin rates by size sign and location sign:')
print(_df.pivot_table(values=outcome_col, index='rel_size_sign', columns='rel_loc_sign', aggfunc='mean'))
