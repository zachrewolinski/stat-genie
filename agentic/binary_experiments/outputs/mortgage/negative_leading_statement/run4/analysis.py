import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.proportion import proportions_ztest

# Load data
path = "mortgage.csv"
df = pd.read_csv(path)

# Basic sanity: keep required columns
cols = [
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
    "deny",
]

# Drop rows with missing values in model columns
model_df = df[cols].dropna().copy()

# Descriptive denial rates by gender
rate_by_gender = model_df.groupby("female")["deny"].mean()
count_by_gender = model_df.groupby("female")["deny"].count()

# Two-proportion z-test (female vs male)
# female==1, male==0
n_female = int(count_by_gender.get(1.0, 0))
n_male = int(count_by_gender.get(0.0, 0))

x_female = int(model_df.loc[model_df["female"] == 1.0, "deny"].sum())
x_male = int(model_df.loc[model_df["female"] == 0.0, "deny"].sum())

stat, pval = proportions_ztest([x_female, x_male], [n_female, n_male])

# Logistic regression with controls
X = model_df.drop(columns=["deny"])
X = sm.add_constant(X)
y = model_df["deny"]

logit = sm.Logit(y, X)
res = logit.fit(disp=False)

# Extract female effect
female_coef = res.params.get("female")
female_p = res.pvalues.get("female")

print("Denial rates by gender (female=1, male=0):")
print(rate_by_gender)
print("Counts by gender:")
print(count_by_gender)
print("Two-proportion z-test p-value (female vs male denial rate):", pval)
print("Logit female coefficient:", female_coef)
print("Logit female p-value:", female_p)
