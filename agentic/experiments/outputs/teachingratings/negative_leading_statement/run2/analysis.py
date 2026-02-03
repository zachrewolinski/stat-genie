import pandas as pd
import statsmodels.formula.api as smf

# Load data
_df = pd.read_csv("teachingratings.csv")

# Basic summary
summary = {
    "n_rows": len(_df),
    "n_prof": _df["prof"].nunique(),
    "eval_mean": _df["eval"].mean(),
    "beauty_mean": _df["beauty"].mean(),
}
print("Summary:", summary)

# Simple correlation
corr = _df[["eval", "beauty"]].corr().loc["eval", "beauty"]
print("Correlation eval-beauty:", corr)

# Simple OLS
model_simple = smf.ols("eval ~ beauty", data=_df).fit()
print("\nSimple OLS eval ~ beauty")
print(model_simple.summary().tables[1])

# Multiple controls based on common covariates
# Use categorical controls and numeric controls
formula = "eval ~ beauty + age + students + C(gender) + C(minority) + C(credits) + C(division) + C(native) + C(tenure)"
model_controls = smf.ols(formula, data=_df).fit()
print("\nOLS with controls")
print(model_controls.summary().tables[1])

# Effect size for beauty in controlled model
coef = model_controls.params.get("beauty")
pval = model_controls.pvalues.get("beauty")
print("\nControlled model beauty coef:", coef, "p-value:", pval)

# Save key results for downstream use
results = {
    "corr_eval_beauty": corr,
    "simple_coef": model_simple.params.get("beauty"),
    "simple_pval": model_simple.pvalues.get("beauty"),
    "controlled_coef": coef,
    "controlled_pval": pval,
}
print("Results:", results)
