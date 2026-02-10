import pandas as pd
import statsmodels.api as sm

# Load data
csv_path = "caschools.csv"

df = pd.read_csv(csv_path)

# Construct key variables
# Student-teacher ratio (students per teacher)
df["stratio"] = df["students"] / df["teachers"]

# Overall academic performance: average of reading and math scores
df["score_avg"] = df[["read", "math"]].mean(axis=1)

summary_lines = []

summary_lines.append("Basic relationships between student-teacher ratio and achievement\n")

# Correlations
corr_read = df["stratio"].corr(df["read"])
corr_math = df["stratio"].corr(df["math"])
corr_avg = df["stratio"].corr(df["score_avg"])

summary_lines.append(f"Correlation(stratio, read): {corr_read:.3f}\n")
summary_lines.append(f"Correlation(stratio, math): {corr_math:.3f}\n")
summary_lines.append(f"Correlation(stratio, avg score): {corr_avg:.3f}\n\n")

# Simple OLS: scores on student-teacher ratio
for outcome in ["read", "math", "score_avg"]:
    y = df[outcome]
    X = sm.add_constant(df["stratio"])
    model = sm.OLS(y, X).fit()
    coef = model.params["stratio"]
    pval = model.pvalues["stratio"]
    summary_lines.append(
        f"OLS {outcome} ~ stratio: coef={coef:.3f}, p-value={pval:.4g}, R^2={model.rsquared:.3f}\n"
    )
summary_lines.append("\n")

# Multiple regression controlling for key demographics and resources
controls = ["income", "calworks", "lunch", "english", "computer", "expenditure"]

for outcome in ["read", "math", "score_avg"]:
    y = df[outcome]
    X = df[["stratio"] + controls].copy()
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()
    coef = model.params["stratio"]
    pval = model.pvalues["stratio"]
    summary_lines.append(
        f"OLS {outcome} ~ stratio + controls: coef={coef:.3f}, p-value={pval:.4g}, R^2={model.rsquared:.3f}\n"
    )

# Rough evidence strength metric based on average t-stat across models
import numpy as np

# Collect t-stats for stratio term from all models
stats = []

for outcome in ["read", "math", "score_avg"]:
    # simple model
    y = df[outcome]
    X_simple = sm.add_constant(df["stratio"])
    m_simple = sm.OLS(y, X_simple).fit()
    stats.append(m_simple.tvalues["stratio"])

    # with controls
    X_ctrl = df[["stratio"] + controls].copy()
    X_ctrl = sm.add_constant(X_ctrl)
    m_ctrl = sm.OLS(y, X_ctrl).fit()
    stats.append(m_ctrl.tvalues["stratio"])

avg_t = float(np.mean(stats))
summary_lines.append(f"\nAverage t-statistic for stratio across models: {avg_t:.3f}\n")

# Map evidence to a Likert-like strength score in [-100, 100].
# Negative t-stat (higher ratio -> lower scores) should yield a positive Likert value,
# because the research question asks whether LOWER ratio is associated with HIGHER scores.

# Transform: strong negative t (e.g., -10 or below) -> +100,
# near-zero t -> 0, positive t (wrong direction) -> negative score.

# Clip t to [-10, 10] for stability
clipped_t = max(min(avg_t, 10.0), -10.0)

# Flip sign so that negative t (desirable direction) becomes positive evidence
flipped = -clipped_t

# Scale linearly from [-10, 10] to [-100, 100]
raw_score = (flipped / 10.0) * 100.0

# Clip and round to nearest integer
likert_score = int(round(max(min(raw_score, 100.0), -100.0)))

summary_lines.append(f"Derived Likert-style evidence score: {likert_score}\n")

# Write a brief human-readable summary for debugging / transparency
with open("analysis_summary.txt", "w") as f:
    f.writelines(summary_lines)

# Also write the scalar conclusion expected by the harness
with open("conclusion.txt", "w") as f:
    f.write(str(likert_score))

