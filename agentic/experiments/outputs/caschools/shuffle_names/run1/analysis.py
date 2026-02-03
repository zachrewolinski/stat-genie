import pandas as pd
import statsmodels.api as sm

# Load data
_df = pd.read_csv("caschools.csv")

# Map shuffled columns to their actual meanings based on value ranges/types
students_total = _df["english"]      # total enrollment
teachers_count = _df["students"]     # number of teachers (FTE)
read_score = _df["district"]         # average reading score
math_score = _df["expenditure"]      # average math score

# Student-teacher ratio and overall academic performance
student_teacher_ratio = students_total / teachers_count
avg_score = (read_score + math_score) / 2

# Correlation
corr = student_teacher_ratio.corr(avg_score)

# Simple linear regression: performance on student-teacher ratio
X = sm.add_constant(student_teacher_ratio)
model = sm.OLS(avg_score, X).fit()

print("Student-teacher ratio summary:")
print(student_teacher_ratio.describe())
print("\nAverage score summary:")
print(avg_score.describe())
print(f"\nPearson correlation (ratio vs. avg score): {corr:.4f}")
print("\nOLS: avg_score ~ student_teacher_ratio")
print(f"Slope (ratio): {model.params.iloc[1]:.4f}")
print(f"p-value (ratio): {model.pvalues.iloc[1]:.6f}")
print(f"R-squared: {model.rsquared:.4f}")
