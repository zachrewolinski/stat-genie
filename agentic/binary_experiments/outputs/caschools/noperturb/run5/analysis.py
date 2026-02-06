import pandas as pd
import numpy as np
import statsmodels.api as sm

# Load data
DATA_PATH = "caschools.csv"
df = pd.read_csv(DATA_PATH)

# Compute student-teacher ratio and average test score
ratio = df["students"] / df["teachers"]
score = df[["read", "math"]].mean(axis=1)

# Basic association: correlation and simple regression
corr = ratio.corr(score)
X_simple = sm.add_constant(ratio)
model_simple = sm.OLS(score, X_simple).fit()

# Regression with common controls to check robustness
X_controls = pd.DataFrame({
    "ratio": ratio,
    "income": df["income"],
    "lunch": df["lunch"],
    "english": df["english"],
    "expenditure": df["expenditure"],
})
X_controls = sm.add_constant(X_controls)
model_controls = sm.OLS(score, X_controls).fit()

# Print key results for transparency
print("Correlation (student-teacher ratio vs avg score):", corr)
print("Simple regression coef on ratio:", model_simple.params.iloc[1], "p=", model_simple.pvalues.iloc[1])
print("Controlled regression coef on ratio:", model_controls.params["ratio"], "p=", model_controls.pvalues["ratio"])

# Save a small summary for reference
summary = {
    "corr_ratio_score": corr,
    "simple_coef": model_simple.params.iloc[1],
    "simple_p": model_simple.pvalues.iloc[1],
    "controls_coef": model_controls.params["ratio"],
    "controls_p": model_controls.pvalues["ratio"],
}
summary_df = pd.DataFrame([summary])
summary_df.to_csv("analysis_summary.csv", index=False)
