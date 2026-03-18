import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats

# Load data
_df = pd.read_csv('affairs.csv')

print('columns', list(_df.columns))
print(_df.head())
print(_df.dtypes)

# identify candidate columns for children and affairs
# Use heuristic: children likely binary or small integer; affairs likely count or non-negative numeric

summary = _df.describe(include='all')
print(summary)

# compute relationship for each column with a likely affairs outcome
# Let's try to identify outcome column by looking for non-negative numeric with many zeros.

numeric_cols = _df.select_dtypes(include=[np.number]).columns.tolist()

# For each numeric col, compute proportion of zeros and range
col_info = []
for c in numeric_cols:
    s = _df[c]
    col_info.append({
        'col': c,
        'min': float(s.min()),
        'max': float(s.max()),
        'mean': float(s.mean()),
        'zero_rate': float((s == 0).mean()),
        'unique': int(s.nunique()),
    })

col_info_sorted = sorted(col_info, key=lambda x: (-x['zero_rate'], -x['max']))
print('numeric col info:', col_info_sorted)

# For the research question, we need children vs affairs.
# We'll assume the column named 'children' is children, and 'affairs' is outcome if exists; otherwise identify.

children_col = 'children' if 'children' in _df.columns else None

if 'affairs' in _df.columns:
    affairs_col = 'affairs'
else:
    # pick numeric column with highest zero rate and reasonable max (>0)
    affairs_col = col_info_sorted[0]['col'] if col_info_sorted else None

print('children_col', children_col, 'affairs_col', affairs_col)

# Basic association tests
if children_col and affairs_col:
    children = _df[children_col]
    affairs = _df[affairs_col]

    # treat children as binary if it looks binary, else numeric
    unique_children = sorted(children.dropna().unique())
    print('unique children values', unique_children[:10], '... total', len(unique_children))

    # correlation (Spearman and Pearson)
    pearson = stats.pearsonr(children, affairs)
    spearman = stats.spearmanr(children, affairs)
    print('pearson', pearson)
    print('spearman', spearman)

    # if children is binary/categorical, do t-test / Mann-Whitney
    if len(unique_children) <= 2:
        grp0 = affairs[children == unique_children[0]]
        grp1 = affairs[children == unique_children[-1]]
        ttest = stats.ttest_ind(grp0, grp1, equal_var=False)
        mwu = stats.mannwhitneyu(grp0, grp1, alternative='two-sided')
        print('ttest', ttest)
        print('mwu', mwu)

    # regression with controls if available
    # Use simple OLS with available numeric predictors (excluding affairs)
    predictors = [c for c in numeric_cols if c != affairs_col]
    X = _df[predictors]
    X = sm.add_constant(X, has_constant='add')
    model = sm.OLS(affairs, X).fit()
    print(model.summary())

    if children_col in model.params.index:
        print('children coef', model.params[children_col], 'p', model.pvalues[children_col])
