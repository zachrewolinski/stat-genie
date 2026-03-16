import json
import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats

# Load data
csv_path = 'teachingratings.csv'
info_path = 'info.json'

with open(info_path, 'r') as f:
    info = json.load(f)

# Read CSV
_df = pd.read_csv(csv_path)

# Basic info for mapping
summary = {}
summary['columns'] = list(_df.columns)
summary['head'] = _df.head(5).to_dict(orient='records')
summary['dtypes'] = _df.dtypes.astype(str).to_dict()
summary['uniques'] = {c: _df[c].dropna().unique()[:10].tolist() for c in _df.columns}

# Identify likely evaluation score column:
# Look for numeric column with range roughly 1-5
numeric_cols = [c for c in _df.columns if pd.api.types.is_numeric_dtype(_df[c])]
range_candidates = []
for c in numeric_cols:
    rng = (_df[c].min(), _df[c].max())
    if rng[0] >= 1 and rng[1] <= 5:
        range_candidates.append((c, rng, _df[c].mean(), _df[c].std()))
summary['rating_candidates'] = range_candidates

# Identify beauty column: numeric with mean near 0 and range approx -2 to 2
beauty_candidates = []
for c in numeric_cols:
    rng = (_df[c].min(), _df[c].max())
    mean = _df[c].mean()
    if rng[0] < 0 and rng[1] > 0 and abs(mean) < 0.2 and (rng[1]-rng[0]) < 5:
        beauty_candidates.append((c, rng, mean, _df[c].std()))
summary['beauty_candidates'] = beauty_candidates

# Create cleaned dataframe for analysis
# Use identified columns based on metadata and candidate checks.
# In this dataset, 'allstudents' looks like the evaluation score and 'beauty' is beauty.

rating_col = 'allstudents'
beauty_col = 'beauty'

# Convert categorical yes/no/male/female/etc to numeric if used as controls
# Map functions

def map_binary(series):
    if series.dropna().nunique() == 2:
        vals = sorted(series.dropna().unique())
        mapping = {vals[0]: 0, vals[1]: 1}
        return series.map(mapping), mapping
    return series, None

# Prepare data
analysis_df = _df.copy()

# Controls: age, tenure, prof, native, gender, credits, division, rownames, minority, students
# We'll include numeric ones directly; categorical ones map to binary.
control_cols = ['age', 'tenure', 'prof', 'native', 'gender', 'credits', 'division', 'rownames', 'minority', 'students']

binary_mappings = {}
for c in control_cols:
    if c in analysis_df.columns and analysis_df[c].dtype == object:
        analysis_df[c], mapping = map_binary(analysis_df[c])
        if mapping:
            binary_mappings[c] = mapping

# Drop rows with missing values in variables used
model_cols = [rating_col, beauty_col] + [c for c in control_cols if c in analysis_df.columns]
model_df = analysis_df[model_cols].dropna()

# Simple correlation
corr = model_df[[rating_col, beauty_col]].corr().iloc[0,1]

# Simple regression: rating ~ beauty
X_simple = sm.add_constant(model_df[[beauty_col]])
model_simple = sm.OLS(model_df[rating_col], X_simple).fit()

# Multiple regression with controls
X_multi = sm.add_constant(model_df[[beauty_col] + [c for c in control_cols if c in model_df.columns]])
model_multi = sm.OLS(model_df[rating_col], X_multi).fit()

# Also compute standardized effect (beta) for beauty in multi model
# Standardize predictors and outcome
std_df = model_df.copy()
std_df[rating_col] = (std_df[rating_col] - std_df[rating_col].mean()) / std_df[rating_col].std(ddof=0)
for c in [beauty_col] + [c for c in control_cols if c in model_df.columns]:
    std_df[c] = (std_df[c] - std_df[c].mean()) / std_df[c].std(ddof=0)
X_std = sm.add_constant(std_df[[beauty_col] + [c for c in control_cols if c in model_df.columns]])
model_std = sm.OLS(std_df[rating_col], X_std).fit()

results = {
    'n': int(model_df.shape[0]),
    'corr': float(corr),
    'simple_coef': float(model_simple.params[beauty_col]),
    'simple_p': float(model_simple.pvalues[beauty_col]),
    'simple_ci': tuple(model_simple.conf_int().loc[beauty_col].tolist()),
    'multi_coef': float(model_multi.params[beauty_col]),
    'multi_p': float(model_multi.pvalues[beauty_col]),
    'multi_ci': tuple(model_multi.conf_int().loc[beauty_col].tolist()),
    'std_beta': float(model_std.params[beauty_col]),
    'std_p': float(model_std.pvalues[beauty_col]),
    'r2_simple': float(model_simple.rsquared),
    'r2_multi': float(model_multi.rsquared),
    'binary_mappings': binary_mappings,
    'rating_candidates': summary['rating_candidates'],
    'beauty_candidates': summary['beauty_candidates'],
}

with open('analysis_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(json.dumps(results, indent=2))
