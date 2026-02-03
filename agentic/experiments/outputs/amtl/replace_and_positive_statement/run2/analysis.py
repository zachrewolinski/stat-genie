import numpy as np
import pandas as pd
import patsy
import statsmodels.api as sm

# Load data
_df = pd.read_csv("amtl.csv")

# Basic cleaning
_df = _df.copy()
_df["is_human"] = (_df["genus"] == "Homo sapiens").astype(int)

# Drop rows with missing values in relevant fields
_df = _df.dropna(subset=["num_amtl", "sockets", "age", "prob_male", "tooth_class", "is_human"])

# Ensure valid binomial counts
_df = _df[_df["sockets"] >= _df["num_amtl"]]

# Build design matrix
X = patsy.dmatrix(
    "is_human + age + prob_male + C(tooth_class)",
    data=_df,
    return_type="dataframe",
)

# Endog as successes/failures
endog = np.column_stack([_df["num_amtl"].to_numpy(), (_df["sockets"] - _df["num_amtl"]).to_numpy()])

model = sm.GLM(endog, X, family=sm.families.Binomial()).fit()

coef = model.params.get("is_human", float("nan"))
pval = model.pvalues.get("is_human", float("nan"))

print(model.summary())
print("is_human coef:", coef)
print("is_human p-value:", pval)
