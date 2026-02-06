import pandas as pd
import statsmodels.formula.api as smf

# Load data
csv_path = "teachingratings.csv"
df = pd.read_csv(csv_path)

# Basic cleaning: ensure categorical fields are treated as such
categorical_cols = ["minority", "gender", "credits", "division", "native", "tenure"]
for col in categorical_cols:
    if col in df.columns:
        df[col] = df[col].astype("category")

# Model 1: bivariate relationship
model1 = smf.ols("eval ~ beauty", data=df).fit(cov_type="HC3")

# Model 2: add controls
controls = "age + C(gender) + C(minority) + C(credits) + C(division) + C(native) + C(tenure) + students + allstudents"
model2 = smf.ols(f"eval ~ beauty + {controls}", data=df).fit(cov_type="HC3")

# Collect key results
results = {
    "model1_coef": model1.params.get("beauty"),
    "model1_p": model1.pvalues.get("beauty"),
    "model2_coef": model2.params.get("beauty"),
    "model2_p": model2.pvalues.get("beauty"),
    "n": int(model2.nobs),
}

print("Bivariate model: coef=%.4f, p=%.4g" % (results["model1_coef"], results["model1_p"]))
print("Controlled model: coef=%.4f, p=%.4g" % (results["model2_coef"], results["model2_p"]))
print("N=", results["n"])
