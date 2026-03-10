import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

DATA_PATH = "teachingratings.csv"

df = pd.read_csv(DATA_PATH)

print("columns", df.columns.tolist())
print("shape", df.shape)
print(df.head())

# basic clean: ensure eval and beauty numeric
for col in ["eval", "beauty"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# compute simple correlation
corr = df[["beauty", "eval"]].corr(method="pearson").iloc[0,1]
print("pearson_corr", corr)

# simple OLS
simple = smf.ols("eval ~ beauty", data=df).fit()
print("simple_ols\n", simple.summary())

# build multivariate model with available covariates typical in dataset
# candidate covariates if present
covariates = []
for c in ["age", "gender", "minority", "native", "tenure", "division", "credits", "students", "allstudents"]:
    if c in df.columns:
        covariates.append(c)

# handle categoricals using C()
formula_terms = []
for c in covariates:
    if df[c].dtype == "object":
        formula_terms.append(f"C({c})")
    else:
        formula_terms.append(c)

formula = "eval ~ beauty"
if formula_terms:
    formula += " + " + " + ".join(formula_terms)

multi = smf.ols(formula, data=df).fit()
print("multi_formula", formula)
print("multi_ols\n", multi.summary())

# standardized effect (beta) for beauty in multivariate
# compute z-scores for numeric variables
zdf = df.copy()
for c in ["eval", "beauty"] + [c for c in covariates if df[c].dtype != "object"]:
    zdf[c] = (pd.to_numeric(zdf[c], errors="coerce") - pd.to_numeric(zdf[c], errors="coerce").mean()) / pd.to_numeric(zdf[c], errors="coerce").std(ddof=0)

z_formula = "eval ~ beauty"
if formula_terms:
    # same terms but for numeric vars already z; categorical unchanged
    z_formula = "eval ~ beauty"
    if formula_terms:
        z_formula += " + " + " + ".join(formula_terms)

z_multi = smf.ols(z_formula, data=zdf).fit()
print("z_multi_formula", z_formula)
print("z_beta_beauty", z_multi.params.get("beauty"), "p", z_multi.pvalues.get("beauty"))

