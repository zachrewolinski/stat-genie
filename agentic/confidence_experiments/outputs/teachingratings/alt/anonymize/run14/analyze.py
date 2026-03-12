import json
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
DATA_PATH = "teachingratings.csv"

df = pd.read_csv(DATA_PATH)

# Rename for clarity
beauty = "feature6"
ratings = "feature7"

# Basic summary
summary = {
    "n_rows": int(df.shape[0]),
    "n_cols": int(df.shape[1]),
    "beauty_mean": float(df[beauty].mean()),
    "beauty_std": float(df[beauty].std()),
    "ratings_mean": float(df[ratings].mean()),
    "ratings_std": float(df[ratings].std()),
}

# Pearson correlation
corr = df[[beauty, ratings]].corr().iloc[0, 1]

# Simple OLS
simple_model = smf.ols(f"{ratings} ~ {beauty}", data=df).fit()

# Full model with controls
# Treat categorical features as categorical using C()
controls = [
    "C(feature2)",  # minority
    "feature3",     # age
    "C(feature4)",  # gender
    "C(feature5)",  # single-credit elective
    "C(feature8)",  # upper/lower
    "C(feature9)",  # native English
    "C(feature10)", # tenure track
    "feature11",    # students participated
    "feature12",    # students enrolled
]
formula = f"{ratings} ~ {beauty} + " + " + ".join(controls)
full_model = smf.ols(formula, data=df).fit()

results = {
    "summary": summary,
    "correlation": float(corr),
    "simple_model": {
        "coef_beauty": float(simple_model.params[beauty]),
        "p_value_beauty": float(simple_model.pvalues[beauty]),
        "r2": float(simple_model.rsquared),
        "ci_lower": float(simple_model.conf_int().loc[beauty, 0]),
        "ci_upper": float(simple_model.conf_int().loc[beauty, 1]),
    },
    "full_model": {
        "coef_beauty": float(full_model.params[beauty]),
        "p_value_beauty": float(full_model.pvalues[beauty]),
        "r2": float(full_model.rsquared),
        "ci_lower": float(full_model.conf_int().loc[beauty, 0]),
        "ci_upper": float(full_model.conf_int().loc[beauty, 1]),
    },
}

with open("analysis_results.json", "w") as f:
    json.dump(results, f, indent=2)

print(json.dumps(results, indent=2))
