import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats

# Load data
path = "teachingratings.csv"
df = pd.read_csv(path)

# Basic variables
beauty = df["feature6"]
rating = df["feature7"]

# Bivariate correlation
corr_r, corr_p = stats.pearsonr(beauty, rating)

# Bivariate regression
X_biv = sm.add_constant(beauty)
model_biv = sm.OLS(rating, X_biv).fit()

# Multivariate regression with controls (exclude feature1 id, feature13 instructor id)
# Categorical controls: feature2, feature4, feature5, feature8, feature9, feature10
cat_cols = ["feature2", "feature4", "feature5", "feature8", "feature9", "feature10"]
num_cols = ["feature3", "feature11", "feature12"]

X = df[["feature6"] + num_cols + cat_cols].copy()

# Create dummies for categorical variables
X = pd.get_dummies(X, columns=cat_cols, drop_first=True)

X = sm.add_constant(X)

model = sm.OLS(rating, X).fit(cov_type="cluster", cov_kwds={"groups": df["feature13"]})

coef = model.params["feature6"]
se = model.bse["feature6"]
pval = model.pvalues["feature6"]
ci_low, ci_high = model.conf_int().loc["feature6"].tolist()

# Standardized effect (per 1 SD of beauty, in SDs of rating)
std_beauty = beauty.std(ddof=1)
std_rating = rating.std(ddof=1)
std_effect = coef * std_beauty / std_rating

# Effect in rating points for 1 SD beauty
points_per_sd = coef * std_beauty

# R-squared
r2 = model.rsquared

# Print summary of key stats
print("n=", len(df))
print("Pearson r=", corr_r, "p=", corr_p)
print("Bivariate coef=", model_biv.params["feature6"], "p=", model_biv.pvalues["feature6"], "R2=", model_biv.rsquared)
print("Multivariate coef=", coef, "SE=", se, "p=", pval, "CI=", (ci_low, ci_high))
print("Standardized effect=", std_effect)
print("Points per 1 SD beauty=", points_per_sd)
print("Multivariate R2=", r2)

# Also check simple group split: top vs bottom beauty quartile
q1 = beauty.quantile(0.25)
q3 = beauty.quantile(0.75)
low = rating[beauty <= q1]
high = rating[beauty >= q3]
print("Top vs bottom quartile mean rating:", high.mean(), low.mean(), "diff=", high.mean()-low.mean())
