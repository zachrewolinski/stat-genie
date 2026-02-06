import pandas as pd
import statsmodels.formula.api as smf

# Load dataset
csv_path = "teachingratings.csv"
df = pd.read_csv(csv_path)

# Basic cleaning: ensure categorical variables treated as such
cat_cols = ["minority", "gender", "credits", "division", "native", "tenure"]
for col in cat_cols:
    df[col] = df[col].astype("category")

# Model 1: simple association
# Simple association (unweighted)
model_simple = smf.ols("eval ~ beauty", data=df).fit()

# Model 2: with controls
# With common controls (unweighted)
model_controls = smf.ols(
    "eval ~ beauty + age + C(gender) + C(minority) + C(native) + C(tenure) + C(division) + C(credits) + students",
    data=df,
).fit()

# Weighted by number of students to reflect evaluation reliability
model_weighted = smf.wls(
    "eval ~ beauty + age + C(gender) + C(minority) + C(native) + C(tenure) + C(division) + C(credits) + students",
    data=df,
    weights=df["students"],
).fit()

print("Simple model coefficient (beauty):", model_simple.params["beauty"], "p=", model_simple.pvalues["beauty"])
print("Controlled model coefficient (beauty):", model_controls.params["beauty"], "p=", model_controls.pvalues["beauty"])
print("Weighted controlled model coefficient (beauty):", model_weighted.params["beauty"], "p=", model_weighted.pvalues["beauty"])
