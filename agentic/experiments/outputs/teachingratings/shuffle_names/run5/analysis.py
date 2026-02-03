import pandas as pd
import statsmodels.formula.api as smf

# Load data
DF_PATH = "teachingratings.csv"
df = pd.read_csv(DF_PATH)

# Basic correlation
corr = df["beauty"].corr(df["allstudents"])

# Simple OLS
model_simple = smf.ols("allstudents ~ beauty", data=df).fit()

# Multiple regression with available covariates
# Treat categorical vars as categories
categoricals = ["eval", "tenure", "prof", "native", "gender", "credits"]
for c in categoricals:
    df[c] = df[c].astype("category")

formula = "allstudents ~ beauty + age + rownames + minority + students + C(eval) + C(tenure) + C(prof) + C(native) + C(gender) + C(credits)"
model_full = smf.ols(formula, data=df).fit()

# Save summary stats to a text file for inspection if needed
with open("analysis_summary.txt", "w") as f:
    f.write("Correlation (beauty, allstudents): {:.4f}\n".format(corr))
    f.write("\nSimple OLS:\n")
    f.write(model_simple.summary().as_text())
    f.write("\n\nFull OLS:\n")
    f.write(model_full.summary().as_text())

# Print key results
print("Correlation:", corr)
print("Simple OLS beauty coef:", model_simple.params["beauty"], "p=", model_simple.pvalues["beauty"])
print("Full OLS beauty coef:", model_full.params["beauty"], "p=", model_full.pvalues["beauty"])
