import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

# Load data
path = "teachingratings.csv"
df = pd.read_csv(path)

# Basic cleaning: ensure categorical columns are treated as such
categorical_cols = ["minority", "gender", "credits", "division", "native", "tenure"]
for col in categorical_cols:
    df[col] = df[col].astype("category")

# Baseline model: eval ~ beauty
model1 = smf.ols("eval ~ beauty", data=df).fit()

# Controlled model with common covariates
# Use log(students) to reduce skew; add age and categorical controls
# Avoid including both students and allstudents to reduce multicollinearity
model2 = smf.ols(
    "eval ~ beauty + age + np.log(students) + C(gender) + C(minority) + C(credits) + C(division) + C(native) + C(tenure)",
    data=df,
).fit()

# Collect results
results = {
    "model1_coef": model1.params.get("beauty", np.nan),
    "model1_p": model1.pvalues.get("beauty", np.nan),
    "model2_coef": model2.params.get("beauty", np.nan),
    "model2_p": model2.pvalues.get("beauty", np.nan),
}

print("Baseline model (eval ~ beauty):")
print(model1.summary().tables[1])
print("\nControlled model:")
print(model2.summary().tables[1])
print("\nKey results:")
for k, v in results.items():
    print(f"{k}: {v}")

# Save key results for downstream use
pd.Series(results).to_csv("analysis_results.csv")
