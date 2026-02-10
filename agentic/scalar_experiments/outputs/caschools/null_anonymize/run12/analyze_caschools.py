import pandas as pd
import statsmodels.api as sm


# Load data
df = pd.read_csv("caschools.csv")

# Construct variables
# Student-teacher ratio: enrollment / teachers
df["stratio"] = df["feature6"] / df["feature7"]

# Academic performance: average of reading and math scores
df["avg_score"] = (df["feature14"] + df["feature15"]) / 2.0

# Basic correlation
corr = df["stratio"].corr(df["avg_score"])

# Regression with controls for socioeconomic and resource variables
X = df[
    [
        "stratio",
        "feature8",
        "feature9",
        "feature10",
        "feature11",
        "feature12",
        "feature13",
    ]
]
X = sm.add_constant(X)
y = df["avg_score"]

model = sm.OLS(y, X).fit()

# Extract coefficient and p-value for student-teacher ratio
coef = model.params["stratio"]
pval = model.pvalues["stratio"]

print("Correlation (stratio vs avg_score):", corr)
print("OLS coef for stratio:", coef)
print("p-value for stratio:", pval)

# Interpret direction: lower ratio -> higher performance means
# coefficient on stratio should be negative (since higher stratio=worse)

if pval < 0.001:
    strength = "very_strong"
elif pval < 0.01:
    strength = "strong"
elif pval < 0.05:
    strength = "moderate"
elif pval < 0.1:
    strength = "weak"
else:
    strength = "none"

print("Significance strength:", strength)
