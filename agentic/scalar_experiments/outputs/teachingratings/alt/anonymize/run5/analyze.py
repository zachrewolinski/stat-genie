import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

# Load data
path = 'teachingratings.csv'
df = pd.read_csv(path)

# Map features based on info.json
beauty = 'feature6'
rating = 'feature7'

# Basic stats
n = len(df)

# Pearson correlation
pearson_r, pearson_p = stats.pearsonr(df[beauty], df[rating])

# Spearman correlation (robust to monotonic)
spearman_r, spearman_p = stats.spearmanr(df[beauty], df[rating])

# Simple OLS: rating ~ beauty
model_simple = smf.ols(f"{rating} ~ {beauty}", data=df).fit()

# More complete model with available controls (basic)
# controls: age (feature3), gender (feature4), minority (feature2), single-credit (feature5),
# upper/lower (feature8), native English (feature9), tenure (feature10), class size (feature12)
# Also include log enrollment maybe. We'll use raw for simplicity.
# We'll treat categories with C().
formula = (
    f"{rating} ~ {beauty} + feature3 + C(feature4) + C(feature2) + C(feature5) + "
    f"C(feature8) + C(feature9) + C(feature10) + feature12"
)
model_controls = smf.ols(formula, data=df).fit()

# Save summary stats to file for later use
summary = {
    'n': n,
    'pearson_r': pearson_r,
    'pearson_p': pearson_p,
    'spearman_r': spearman_r,
    'spearman_p': spearman_p,
    'simple_coef': model_simple.params[beauty],
    'simple_p': model_simple.pvalues[beauty],
    'simple_r2': model_simple.rsquared,
    'controls_coef': model_controls.params[beauty],
    'controls_p': model_controls.pvalues[beauty],
    'controls_r2': model_controls.rsquared,
}

# write to json for inspection
import json
with open('analysis_results.json', 'w') as f:
    json.dump(summary, f, indent=2)
