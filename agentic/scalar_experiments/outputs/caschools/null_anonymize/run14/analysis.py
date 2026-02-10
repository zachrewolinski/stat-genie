import pandas as pd
import numpy as np
import statsmodels.api as sm

# Load data
csv_path = "caschools.csv"
df = pd.read_csv(csv_path)

# Define variables based on info.json documentation
# feature6: total enrollment
# feature7: number of teachers
# feature14: average reading score
# feature15: average math score

df = df.copy()

# Compute student-teacher ratio (students per teacher)
df["stratio"] = df["feature6"] / df["feature7"]

# Academic performance: average of reading and math scores
df["testscr"] = (df["feature14"] + df["feature15"]) / 2.0

# Drop any rows with missing values in key columns (defensive, though none expected)
key_cols = ["stratio", "testscr"]
df_clean = df.dropna(subset=key_cols)

n = len(df_clean)

# Basic descriptive statistics
str_mean = df_clean["stratio"].mean()
str_std = df_clean["stratio"].std()
test_mean = df_clean["testscr"].mean()
test_std = df_clean["testscr"].std()

# Pearson correlation between student-teacher ratio and test scores
corr = df_clean["stratio"].corr(df_clean["testscr"])

# Simple OLS regression: testscr ~ stratio
X = sm.add_constant(df_clean["stratio"])
model = sm.OLS(df_clean["testscr"], X).fit()
coef = model.params["stratio"]
se = model.bse["stratio"]
t_value = model.tvalues["stratio"]
p_value = model.pvalues["stratio"]

# Effect size: change in testscr per 1-student increase in ratio
# Also compute standardized beta using correlation and SD ratio
effect_per_student = coef
std_beta = corr * (test_std / str_std) if str_std != 0 else np.nan

# Summarize strength of evidence and association for mapping to Likert
# We will output a small summary file for human-readable inspection
summary_lines = []
summary_lines.append(f"N districts: {n}")
summary_lines.append(f"Mean student-teacher ratio: {str_mean:.2f} (SD {str_std:.2f})")
summary_lines.append(f"Mean test score: {test_mean:.2f} (SD {test_std:.2f})")
summary_lines.append(f"Correlation(stratio, testscr): {corr:.3f}")
summary_lines.append("OLS: testscr = alpha + beta*stratio")
summary_lines.append(f"beta (per +1 student per teacher): {coef:.3f}")
summary_lines.append(f"t-value: {t_value:.2f}, p-value: {p_value:.3g}")
summary_lines.append(f"Standardized beta (approx): {std_beta:.3f}")

# Very rough rule to map to Likert strength (saved for inspection only).
# The actual scalar will be chosen manually based on these diagnostics.
with open("analysis_summary.txt", "w") as f:
    for line in summary_lines:
        f.write(line + "\n")

print("Analysis complete. Summary written to analysis_summary.txt")
