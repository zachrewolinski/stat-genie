import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
df = pd.read_csv("boxes.csv")

# Binary indicator for majority choice
df["majority_choice"] = (df["y"] == 2).astype(int)

# Basic checks
n = len(df)
majority_rate = df["majority_choice"].mean()
minority_rate = (df["y"] == 3).mean()
third_rate = (df["y"] == 1).mean()

# Majority by age (treat age as categorical for simplicity)
age_majority = df.groupby("age")["majority_choice"].mean()

# Majority by culture
culture_majority = df.groupby("culture")["majority_choice"].mean()

print("N =", n)
print("Overall majority choice rate:", majority_rate)
print("Overall minority choice rate:", minority_rate)
print("Overall third-option rate:", third_rate)
print("\nMajority rate by age:")
print(age_majority)
print("\nMajority rate by culture:")
print(culture_majority)

# Simple measure of variation across cultures and ages
age_var = age_majority.var()
cul_var = culture_majority.var()
print("\nVariance of majority rate by age:", age_var)
print("Variance of majority rate by culture:", cul_var)

# Logistic regression: majority choice ~ age + culture
model = smf.glm(
    formula="majority_choice ~ age + C(culture)",
    data=df,
    family=sm.families.Binomial(),
).fit()

print("\nLogistic regression summary (truncated):")
print(model.summary2().tables[1])

# Pseudo R-squared as rough effect-size summary
null_llf = model.null_deviance / -2.0
model_llf = model.deviance / -2.0
pseudo_r2 = 1 - model_llf / null_llf
print("\nApprox. McFadden pseudo R^2:", pseudo_r2)
