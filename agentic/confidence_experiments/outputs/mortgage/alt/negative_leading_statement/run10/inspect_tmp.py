import pandas as pd
import numpy as np
import statsmodels.api as sm


df = pd.read_csv("mortgage.csv")

binary_cols = ["female", "black", "self_employed", "married", "bad_history", "deny", "denied_PMI", "accept"]
for col in binary_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

analysis_df = df.dropna(subset=["female", "accept"]).copy()
control_cols = [
    "black",
    "housing_expense_ratio",
    "self_employed",
    "married",
    "mortgage_credit",
    "consumer_credit",
    "bad_history",
    "PI_ratio",
    "loan_to_value",
    "denied_PMI",
]
model_cols = ["accept", "female"] + [c for c in control_cols if c in analysis_df.columns]
model_df = analysis_df[model_cols].dropna().copy()

X = sm.add_constant(model_df.drop(columns=["accept"]), has_constant="add")
Y = model_df["accept"]

result = sm.Logit(Y, X).fit(disp=False)
print(result.summary())

# Compute average marginal effects
marg = result.get_margeff(at="overall")
print(marg.summary())
