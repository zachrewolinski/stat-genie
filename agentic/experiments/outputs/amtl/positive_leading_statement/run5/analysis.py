import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
import patsy

# Load data
DF = pd.read_csv("amtl.csv")

# Basic cleaning
DF = DF.copy()
DF = DF[DF["sockets"] > 0]
DF = DF[(DF["num_amtl"] >= 0) & (DF["num_amtl"] <= DF["sockets"])]

# Binary indicator for Homo sapiens
DF["is_homo"] = (DF["genus"] == "Homo sapiens").astype(int)

# Ensure categorical
DF["tooth_class"] = DF["tooth_class"].astype("category")

# Fit binomial GLM with grouped binomial response
# Controls: age, prob_male, tooth_class
y = np.column_stack([DF["num_amtl"], DF["sockets"] - DF["num_amtl"]])
X = patsy.dmatrix(
    "is_homo + age + prob_male + C(tooth_class)",
    DF,
    return_type="dataframe",
)
model = sm.GLM(y, X, family=sm.families.Binomial()).fit()

# Extract coefficient for Homo sapiens indicator
coef = model.params["is_homo"]
pval = model.pvalues["is_homo"]

# Also compute predicted AMTL rate at mean covariates for homo vs non-homo
mean_age = DF["age"].mean()
mean_prob_male = DF["prob_male"].mean()
# Use reference tooth_class (first category)
ref_tooth = DF["tooth_class"].cat.categories[0]

pred_df = pd.DataFrame({
    "is_homo": [0, 1],
    "age": [mean_age, mean_age],
    "prob_male": [mean_prob_male, mean_prob_male],
    "tooth_class": [ref_tooth, ref_tooth],
})

design_info = X.design_info
pred_X = patsy.build_design_matrices([design_info], pred_df, return_type="dataframe")[0]
pred = model.predict(pred_X)

# Decision rule: higher AMTL for Homo sapiens and statistically significant
conclusion_yes = (coef > 0) and (pval < 0.05)

with open("conclusion.txt", "w") as f:
    f.write("Yes\n" if conclusion_yes else "No\n")
    f.write(
        f"GLM (binomial) controlling for age, sex, and tooth class shows a "
        f"{'higher' if coef > 0 else 'lower'} AMTL rate for Homo sapiens "
        f"(coef={coef:.3f}, p={pval:.3g}); "
        f"predicted rate at mean covariates is {pred[1]:.3f} vs {pred[0]:.3f}.\n"
    )

# Save a brief text report for transparency
with open("analysis_report.txt", "w") as f:
    f.write(model.summary().as_text())
    f.write("\n\n")
    f.write(f"Homo sapiens coef: {coef:.6f}, p-value: {pval:.6g}\n")
    f.write(f"Predicted rate (non-homo, homo): {pred[0]:.6f}, {pred[1]:.6f}\n")

print("Model fit complete.")
print(f"Homo sapiens coef: {coef:.6f}, p-value: {pval:.6g}")
print(f"Predicted rate (non-homo, homo): {pred[0]:.6f}, {pred[1]:.6f}")
