import json
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf

# Load data
df = pd.read_csv('affairs.csv')

# Map columns based on info.json descriptions
children_col = 'religiousness'  # description indicates "Are there children in the marriage?"
affairs_col = 'age'            # description indicates frequency of extramarital intercourse

# Clean data
use_df = df[[children_col, affairs_col, 'gender', 'education', 'occupation', 'children', 'rating', 'yearsmarried', 'rownames', 'affairs']].copy()
use_df = use_df.dropna(subset=[children_col, affairs_col])

# Ensure children variable is categorical
use_df[children_col] = use_df[children_col].astype('category')

# Group stats
groups = {}
for level, g in use_df.groupby(children_col):
    groups[str(level)] = {
        'n': int(g.shape[0]),
        'mean': float(g[affairs_col].mean()),
        'median': float(g[affairs_col].median()),
        'std': float(g[affairs_col].std(ddof=1)),
    }

# Two-group tests (expecting yes/no)
levels = list(use_df[children_col].cat.categories)
if len(levels) == 2:
    g0 = use_df[use_df[children_col] == levels[0]][affairs_col]
    g1 = use_df[use_df[children_col] == levels[1]][affairs_col]
    # Welch t-test
    t_res = stats.ttest_ind(g0, g1, equal_var=False)
    # Mann-Whitney U
    mw_res = stats.mannwhitneyu(g0, g1, alternative='two-sided')

    # Effect size (Cohen's d using pooled SD)
    n0, n1 = len(g0), len(g1)
    s0, s1 = g0.std(ddof=1), g1.std(ddof=1)
    pooled_sd = np.sqrt(((n0 - 1) * s0**2 + (n1 - 1) * s1**2) / (n0 + n1 - 2))
    d = (g1.mean() - g0.mean()) / pooled_sd if pooled_sd > 0 else np.nan
else:
    t_res = None
    mw_res = None
    d = np.nan

# Regression with controls (OLS)
# Treat children and gender as categorical
formula = f"{affairs_col} ~ C({children_col}) + C(gender) + education + occupation + children + rating + yearsmarried + rownames + affairs"
model = smf.ols(formula, data=use_df).fit(cov_type='HC3')

# Extract the children effect (if present)
children_terms = [k for k in model.params.index if k.startswith(f"C({children_col})")]
children_effects = {k: {'coef': float(model.params[k]), 'p': float(model.pvalues[k])} for k in children_terms}

results = {
    'groups': groups,
    'ttest': {'statistic': float(t_res.statistic), 'pvalue': float(t_res.pvalue)} if t_res is not None else None,
    'mannwhitney': {'statistic': float(mw_res.statistic), 'pvalue': float(mw_res.pvalue)} if mw_res is not None else None,
    'cohens_d': float(d),
    'ols': {
        'r2': float(model.rsquared),
        'children_effects': children_effects,
    }
}

print(json.dumps(results, indent=2))
