import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

DATA_PATH = "teachingratings.csv"

df = pd.read_csv(DATA_PATH)
print("columns", df.columns.tolist())
print("shape", df.shape)
print(df.head())

# ensure numeric
print("missing eval", df['eval'].isna().sum(), "missing beauty", df['beauty'].isna().sum())

# correlation
corr = df[['beauty','eval']].corr().iloc[0,1]
print("corr beauty-eval", corr)

# simple regression
model_simple = smf.ols("eval ~ beauty", data=df).fit()
print(model_simple.summary())

# try with controls if available
# Identify possible controls in dataset
controls = []
for col in ['age','gender','minority','native','tenure','division','credits','students','allstudents']:
    if col in df.columns:
        controls.append(col)
print("controls", controls)

# build formula with categorical for certain controls
cat_cols = set([c for c in ['gender','minority','native','tenure','division','credits'] if c in df.columns])
formula_terms = ['beauty']
for c in controls:
    if c in cat_cols:
        formula_terms.append(f"C({c})")
    else:
        formula_terms.append(c)
formula = "eval ~ " + " + ".join(formula_terms)
print("formula", formula)

model_controls = smf.ols(formula, data=df).fit()
print(model_controls.summary())

# Save key metrics to file for reference
out = {
    "corr": corr,
    "simple_coef": model_simple.params.get('beauty'),
    "simple_p": model_simple.pvalues.get('beauty'),
    "simple_r2": model_simple.rsquared,
    "controls_coef": model_controls.params.get('beauty'),
    "controls_p": model_controls.pvalues.get('beauty'),
    "controls_r2": model_controls.rsquared,
}
print(out)
