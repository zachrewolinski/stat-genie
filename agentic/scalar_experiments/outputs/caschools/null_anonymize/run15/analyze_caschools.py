import pandas as pd
import numpy as np
import statsmodels.api as sm

# Load data
df = pd.read_csv("caschools.csv")

# Map columns based on metadata
# feature6: total enrollment
# feature7: number of teachers (FTE)
# feature14: average reading score
# feature15: average math score

df = df.copy()

# Compute student-teacher ratio: students per teacher
# Guard against division by zero
ratio = df["feature6"] / df["feature7"].replace(0, np.nan)
df["student_teacher_ratio"] = ratio

# Academic performance: mean of reading and math scores
df["avg_score"] = df[["feature14", "feature15"]].mean(axis=1)

# Drop rows with missing key variables
analysis_df = df[
    [
        "student_teacher_ratio",
        "feature14",
        "feature15",
        "avg_score",
        "feature8",
        "feature9",
        "feature10",
        "feature11",
        "feature12",
        "feature13",
    ]
].dropna()

print("N used in analysis:", len(analysis_df))

# Simple Pearson correlations
corr_read = analysis_df["student_teacher_ratio"].corr(analysis_df["feature14"])
corr_math = analysis_df["student_teacher_ratio"].corr(analysis_df["feature15"])
corr_avg = analysis_df["student_teacher_ratio"].corr(analysis_df["avg_score"])

print("Correlation (ratio, reading):", corr_read)
print("Correlation (ratio, math):   ", corr_math)
print("Correlation (ratio, avg):    ", corr_avg)

# OLS regression of avg_score on student_teacher_ratio + controls
X = analysis_df[
    [
        "student_teacher_ratio",
        "feature8",
        "feature9",
        "feature11",
        "feature12",
        "feature13",
    ]
]
X = sm.add_constant(X)
y = analysis_df["avg_score"]

model = sm.OLS(y, X).fit()
print("\nOLS results:")
print(model.summary())

coef_ratio = model.params["student_teacher_ratio"]
t_ratio = model.tvalues["student_teacher_ratio"]
p_ratio = model.pvalues["student_teacher_ratio"]

print("\nCoefficient on student_teacher_ratio:", coef_ratio)
print("t-stat:", t_ratio)
print("p-value:", p_ratio)

# For convenience, also compute standardized effect size (beta) approximately
ratio_std = analysis_df["student_teacher_ratio"].std()
avg_std = analysis_df["avg_score"].std()
std_effect = coef_ratio * ratio_std / avg_std
print("Approx standardized effect (beta):", std_effect)
