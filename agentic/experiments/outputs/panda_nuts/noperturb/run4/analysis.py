import pandas as pd
import statsmodels.formula.api as smf

# Load data
DATA_PATH = "panda_nuts.csv"

df = pd.read_csv(DATA_PATH)

# Define efficiency as nuts opened per second
# Avoid division by zero if any (none expected)
df["efficiency"] = df["nuts_opened"] / df["seconds"]

# Ensure categorical variables
for col in ["sex", "help", "hammer"]:
    df[col] = df[col].astype("category")

# Model: efficiency ~ age + sex + help
# Use cluster-robust SE by chimpanzee to account for repeated measures
model = smf.ols("efficiency ~ age + sex + help", data=df).fit(
    cov_type="cluster", cov_kwds={"groups": df["chimpanzee"]}
)

# Also check model with hammer as a robustness control (not primary)
model_with_hammer = smf.ols("efficiency ~ age + sex + help + hammer", data=df).fit(
    cov_type="cluster", cov_kwds={"groups": df["chimpanzee"]}
)

# Save key outputs for conclusion
summary_df = pd.DataFrame({
    "term": model.params.index,
    "coef": model.params.values,
    "pval": model.pvalues.values,
})

summary_df_hammer = pd.DataFrame({
    "term": model_with_hammer.params.index,
    "coef": model_with_hammer.params.values,
    "pval": model_with_hammer.pvalues.values,
})

summary_df.to_csv("model_primary.csv", index=False)
summary_df_hammer.to_csv("model_with_hammer.csv", index=False)

# Print a brief summary to stdout for inspection
print("Primary model (efficiency ~ age + sex + help) with cluster-robust SEs")
print(model.summary())
print("\nRobustness model (add hammer)")
print(model_with_hammer.summary())
