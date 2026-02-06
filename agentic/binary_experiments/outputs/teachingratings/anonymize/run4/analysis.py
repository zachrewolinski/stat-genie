import pandas as pd
import statsmodels.formula.api as smf

# Load data
_df = pd.read_csv("teachingratings.csv")

# Rename for clarity
_df = _df.rename(
    columns={
        "feature2": "minority",
        "feature3": "age",
        "feature4": "gender",
        "feature5": "single_credit",
        "feature6": "beauty",
        "feature7": "rating",
        "feature8": "upper_div",
        "feature9": "native_english",
        "feature10": "tenure_track",
        "feature11": "n_eval",
        "feature12": "n_enroll",
        "feature13": "instructor_id",
    }
)

# Basic bivariate regression
model_bivar = smf.ols("rating ~ beauty", data=_df).fit(cov_type="HC3")

# Multivariate regression with common controls
formula = (
    "rating ~ beauty + age + C(gender) + C(minority) + C(single_credit) "
    "+ C(upper_div) + C(native_english) + C(tenure_track) + n_eval + n_enroll"
)
model_full = smf.ols(formula, data=_df).fit(cov_type="HC3")

# Output key results
print("Bivariate model (rating ~ beauty)")
print(model_bivar.summary())
print("\nFull model with controls")
print(model_full.summary())

# Extract key stats
coef_bivar = model_bivar.params["beauty"]
pval_bivar = model_bivar.pvalues["beauty"]
coef_full = model_full.params["beauty"]
pval_full = model_full.pvalues["beauty"]

print("\nKey results:")
print(f"Bivariate beauty coef: {coef_bivar:.4f}, p-value: {pval_bivar:.4g}")
print(f"Full model beauty coef: {coef_full:.4f}, p-value: {pval_full:.4g}")
