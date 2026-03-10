import pandas as pd
import statsmodels.formula.api as smf
import numpy as np

# Load data
path = "teachingratings.csv"
df = pd.read_csv(path)

# Rename for clarity
col_map = {
    "feature2": "minority",
    "feature3": "age",
    "feature4": "gender",
    "feature5": "single_credit",
    "feature6": "beauty",
    "feature7": "rating",
    "feature8": "division",
    "feature9": "native_english",
    "feature10": "tenure_track",
    "feature11": "students_eval",
    "feature12": "students_enrolled",
    "feature13": "instructor_id",
}

df = df.rename(columns=col_map)

# Basic correlation
corr = df[["beauty", "rating"]].corr().iloc[0, 1]

# Simple OLS
model_simple = smf.ols("rating ~ beauty", data=df).fit()

# Multiple regression with covariates
# Use categorical dummies for factors
formula = (
    "rating ~ beauty + age + C(gender) + C(minority) + C(single_credit) + "
    "C(division) + C(native_english) + C(tenure_track) + students_eval + students_enrolled"
)
model_full = smf.ols(formula, data=df).fit(cov_type="HC1")

# Cluster-robust SE by instructor (if enough clusters)
model_full_cluster = smf.ols(formula, data=df).fit(cov_type="cluster", cov_kwds={"groups": df["instructor_id"]})

# Standardized effect of beauty (z-score) in full model
# Standardize beauty and rating
beauty_z = (df["beauty"] - df["beauty"].mean()) / df["beauty"].std(ddof=0)
rating_z = (df["rating"] - df["rating"].mean()) / df["rating"].std(ddof=0)
model_std = smf.ols("rating_z ~ beauty_z", data=df.assign(beauty_z=beauty_z, rating_z=rating_z)).fit()

# Collect results
results = {
    "corr": corr,
    "simple_coef": model_simple.params["beauty"],
    "simple_p": model_simple.pvalues["beauty"],
    "simple_ci": model_simple.conf_int().loc["beauty"].tolist(),
    "full_coef": model_full.params["beauty"],
    "full_p": model_full.pvalues["beauty"],
    "full_ci": model_full.conf_int().loc["beauty"].tolist(),
    "full_cluster_coef": model_full_cluster.params["beauty"],
    "full_cluster_p": model_full_cluster.pvalues["beauty"],
    "full_cluster_ci": model_full_cluster.conf_int().loc["beauty"].tolist(),
    "std_coef": model_std.params["beauty_z"],
    "std_p": model_std.pvalues["beauty_z"],
}

print("RESULTS")
for k, v in results.items():
    print(f"{k}: {v}")
