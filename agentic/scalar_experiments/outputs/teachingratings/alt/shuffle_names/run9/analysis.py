import json
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm

# Load data
csv_path = 'teachingratings.csv'
df = pd.read_csv(csv_path)

# Identify key variables based on metadata
beauty_col = 'beauty'
ratings_col = 'allstudents'

# Basic checks
if beauty_col not in df.columns or ratings_col not in df.columns:
    raise ValueError(f"Missing required columns: {beauty_col} or {ratings_col}")

# Drop rows with missing values in key columns
sub = df[[beauty_col, ratings_col]].dropna()

n = len(sub)
beauty = sub[beauty_col].astype(float)
ratings = sub[ratings_col].astype(float)

# Correlation
pearson_r, pearson_p = stats.pearsonr(beauty, ratings)

# Simple OLS
X = sm.add_constant(beauty)
ols_model = sm.OLS(ratings, X).fit(cov_type='HC3')

# Standardized effect (beta * sd_x / sd_y)
sd_beauty = beauty.std(ddof=1)
sd_ratings = ratings.std(ddof=1)
std_beta = (ols_model.params[beauty_col] * sd_beauty) / sd_ratings if sd_ratings != 0 else np.nan

# 95% CI for slope
conf_int = ols_model.conf_int().loc[beauty_col].tolist()

# Also run an adjusted model with other available covariates to test robustness
# Use all other columns as controls, treating non-numeric as categorical
# Exclude obvious identifiers if present (division, students, rownames) to avoid overfitting identifiers.
exclude = {beauty_col, ratings_col, 'division', 'students', 'rownames'}
controls = [c for c in df.columns if c not in exclude]

# Build design matrix for controls
# Convert categorical columns to category dtype to get dummies
control_df = df[controls].copy()
for c in control_df.columns:
    if control_df[c].dtype == 'object':
        control_df[c] = control_df[c].astype('category')

# Combine and drop missing
adj_df = pd.concat([df[[ratings_col, beauty_col]], control_df], axis=1).dropna()

# Create dummies for categorical controls
adj_X = pd.get_dummies(adj_df.drop(columns=[ratings_col]), drop_first=True)
adj_X = sm.add_constant(adj_X)
adj_y = adj_df[ratings_col].astype(float)

adj_model = sm.OLS(adj_y, adj_X).fit(cov_type='HC3')

adj_beta = adj_model.params.get(beauty_col, np.nan)
adj_p = adj_model.pvalues.get(beauty_col, np.nan)
adj_ci = adj_model.conf_int().loc[beauty_col].tolist() if beauty_col in adj_model.params else [np.nan, np.nan]

results = {
    "n": int(n),
    "beauty_mean": float(beauty.mean()),
    "beauty_sd": float(sd_beauty),
    "ratings_mean": float(ratings.mean()),
    "ratings_sd": float(sd_ratings),
    "pearson_r": float(pearson_r),
    "pearson_p": float(pearson_p),
    "ols_slope": float(ols_model.params[beauty_col]),
    "ols_p": float(ols_model.pvalues[beauty_col]),
    "ols_ci_low": float(conf_int[0]),
    "ols_ci_high": float(conf_int[1]),
    "std_beta": float(std_beta),
    "adj_n": int(adj_df.shape[0]),
    "adj_slope": float(adj_beta),
    "adj_p": float(adj_p),
    "adj_ci_low": float(adj_ci[0]),
    "adj_ci_high": float(adj_ci[1]),
}

print(json.dumps(results, indent=2))
