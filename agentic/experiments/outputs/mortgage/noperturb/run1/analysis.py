import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats

DATA_PATH = "mortgage.csv"

df = pd.read_csv(DATA_PATH)

# Drop index column if present
if "Unnamed: 0" in df.columns:
    df = df.drop(columns=["Unnamed: 0"])

# Basic checks
print("Rows, cols:", df.shape)
print("Female distribution:")
print(df["female"].value_counts(dropna=False).sort_index())
print("Accept distribution:")
print(df["accept"].value_counts(dropna=False).sort_index())

# Approval rate by gender
rates = df.groupby("female")["accept"].mean()
counts = df.groupby("female")["accept"].agg(["mean", "count", "sum"]).rename(columns={"sum": "accepted"})
print("\nApproval rates by gender (female=1):")
print(counts)

# Two-proportion z-test
# females
p1 = counts.loc[1, "mean"]
n1 = counts.loc[1, "count"]
# males
p0 = counts.loc[0, "mean"]
n0 = counts.loc[0, "count"]

# Pooled proportion
p_pool = (counts.loc[1, "accepted"] + counts.loc[0, "accepted"]) / (n1 + n0)
se = np.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n0))
z = (p1 - p0) / se
p_value = 2 * (1 - stats.norm.cdf(abs(z)))
print("\nTwo-proportion z-test (female vs male approval rate):")
print("z=", z, "p=", p_value)

# Logistic regression with controls
# Define outcome and predictors
outcome = df["accept"]

predictors = [
    "female",
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

X = df[predictors]
X = sm.add_constant(X)

model = sm.Logit(outcome, X, missing="drop")
result = model.fit(disp=False)
print("\nLogit regression (accept ~ predictors):")
print(result.summary())

# Extract female effect
coef = result.params["female"]
se = result.bse["female"]
pval = result.pvalues["female"]

# Convert to odds ratio
odds_ratio = float(np.exp(coef))
print("\nFemale coefficient:")
print("coef=", coef, "se=", se, "p=", pval, "odds_ratio=", odds_ratio)

# Predicted approval at average covariates for male vs female
mean_vals = X.mean()

# male
mean_male = mean_vals.copy()
mean_male["female"] = 0
# female
mean_female = mean_vals.copy()
mean_female["female"] = 1

pred_male = result.predict(mean_male)
pred_female = result.predict(mean_female)
print("\nPredicted approval at mean covariates:")
print("male:", float(pred_male), "female:", float(pred_female), "diff:", float(pred_female - pred_male))
