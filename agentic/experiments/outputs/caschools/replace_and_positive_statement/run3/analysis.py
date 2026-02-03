import pandas as pd
import statsmodels.api as sm

# Load data
csv_path = "caschools.csv"
df = pd.read_csv(csv_path)

# Compute student-teacher ratio (students per teacher) and average test score
df = df.copy()
df["student_teacher_ratio"] = df["students"] / df["teachers"]
df["avg_score"] = df[["read", "math"]].mean(axis=1)

# Drop rows with missing values in key fields
analysis_df = df.dropna(subset=["student_teacher_ratio", "avg_score"])

# Correlation
corr = analysis_df[["student_teacher_ratio", "avg_score"]].corr().loc[
    "student_teacher_ratio", "avg_score"
]

# Simple linear regression: avg_score ~ student_teacher_ratio
X = sm.add_constant(analysis_df["student_teacher_ratio"])
model = sm.OLS(analysis_df["avg_score"], X).fit()

# Multiple regression with key controls (common in caschools analyses)
controls = ["lunch", "english", "income", "expenditure"]
controls_df = analysis_df.dropna(subset=controls)
X2 = sm.add_constant(controls_df[["student_teacher_ratio"] + controls])
model2 = sm.OLS(controls_df["avg_score"], X2).fit()

print("Correlation (student_teacher_ratio vs avg_score):", corr)
print("\nSimple regression (avg_score ~ student_teacher_ratio)")
print(model.summary())
print(
    "\nMultiple regression with controls "
    "(avg_score ~ student_teacher_ratio + lunch + english + income + expenditure)"
)
print(model2.summary())
