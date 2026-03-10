import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
csv_path = "teachingratings.csv"
df = pd.read_csv(csv_path)

# Map columns for readability
beauty = "feature6"  # instructor beauty rating (mean-centered)
rating = "feature7"  # teaching evaluation score

# Create derived variables
# Response rate (participants/enrollment), guard against divide-by-zero
if (df["feature12"] <= 0).any():
    df["response_rate"] = np.nan
else:
    df["response_rate"] = df["feature11"] / df["feature12"]

# Log enrollment for scale
# add small constant to avoid log(0)
df["log_enrollment"] = np.log(df["feature12"] + 1)

# Encode categorical variables for regression
# Use patsy formula with C() for categorical

# Basic bivariate regression
model_biv = smf.ols(f"{rating} ~ {beauty}", data=df).fit()

# Multivariable regression with common controls
formula = (
    f"{rating} ~ {beauty} + C(feature2) + feature3 + C(feature4) + "
    "C(feature5) + C(feature8) + C(feature9) + C(feature10) + "
    "log_enrollment + response_rate"
)
model_multi = smf.ols(formula, data=df).fit()

# Standardized coefficient for beauty in multivariable model
# Standardize beauty and rating
beauty_std = (df[beauty] - df[beauty].mean()) / df[beauty].std(ddof=0)
rating_std = (df[rating] - df[rating].mean()) / df[rating].std(ddof=0)
df_std = df.copy()
df_std["beauty_std"] = beauty_std
df_std["rating_std"] = rating_std
formula_std = (
    "rating_std ~ beauty_std + C(feature2) + feature3 + C(feature4) + "
    "C(feature5) + C(feature8) + C(feature9) + C(feature10) + "
    "log_enrollment + response_rate"
)
model_std = smf.ols(formula_std, data=df_std).fit()

# Effect size: predicted rating change for +1 SD beauty (in original rating units)
beauty_sd = df[beauty].std(ddof=0)
coef_beauty = model_multi.params[beauty]
# one SD change in beauty -> coef * SD
rating_change_1sd = coef_beauty * beauty_sd

# Build results summary
results = {
    "n": int(df.shape[0]),
    "beauty_mean": float(df[beauty].mean()),
    "beauty_sd": float(beauty_sd),
    "rating_mean": float(df[rating].mean()),
    "rating_sd": float(df[rating].std(ddof=0)),
    "corr_pearson": float(df[[beauty, rating]].corr().iloc[0, 1]),
    "biv_coef": float(model_biv.params[beauty]),
    "biv_p": float(model_biv.pvalues[beauty]),
    "biv_r2": float(model_biv.rsquared),
    "multi_coef": float(coef_beauty),
    "multi_p": float(model_multi.pvalues[beauty]),
    "multi_r2": float(model_multi.rsquared),
    "std_coef": float(model_std.params["beauty_std"]),
    "std_p": float(model_std.pvalues["beauty_std"]),
    "rating_change_1sd": float(rating_change_1sd),
}

# Save for inspection
with open("analysis_results.json", "w") as f:
    json.dump(results, f, indent=2)

print(json.dumps(results, indent=2))
