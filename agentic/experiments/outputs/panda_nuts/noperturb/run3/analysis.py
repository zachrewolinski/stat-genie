import pandas as pd
import statsmodels.formula.api as smf

# Load data
csv_path = "panda_nuts.csv"
df = pd.read_csv(csv_path)

# Compute efficiency: nuts opened per second
# Avoid division by zero just in case
if (df["seconds"] <= 0).any():
    raise ValueError("Non-positive session duration found.")

df["efficiency"] = df["nuts_opened"] / df["seconds"]

# Fit linear model with categorical predictors for sex and help
model = smf.ols("efficiency ~ age + C(sex) + C(help)", data=df).fit()

# Save key results to a csv for traceability
results = pd.DataFrame(
    {
        "term": model.params.index,
        "coef": model.params.values,
        "pvalue": model.pvalues.values,
    }
)
results.to_csv("model_results.csv", index=False)

# Print a concise summary to stdout
print(model.summary())
