import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

# Load data
csv_path = "panda_nuts.csv"
df = pd.read_csv(csv_path)

# Rename columns for clarity
col_map = {
    "feature1": "id",
    "feature2": "age",
    "feature3": "sex",
    "feature4": "hammer_type",
    "feature5": "nuts_opened",
    "feature6": "duration_sec",
    "feature7": "help"
}
df = df.rename(columns=col_map)

# Compute efficiency: nuts opened per second
# Avoid division by zero just in case

df["efficiency"] = df["nuts_opened"] / df["duration_sec"].replace(0, np.nan)

# Drop any rows with missing efficiency
analysis_df = df.dropna(subset=["efficiency", "age", "sex", "help"]).copy()

# Encode categorical variables (sex, help)
analysis_df["sex"] = analysis_df["sex"].astype("category")
analysis_df["help"] = analysis_df["help"].astype("category")

# Fit linear model: efficiency ~ age + sex + help
model = smf.ols("efficiency ~ age + C(sex) + C(help)", data=analysis_df).fit()

# Also try log efficiency to reduce skewness if needed
analysis_df["log_efficiency"] = np.log(analysis_df["efficiency"] + 1e-6)
log_model = smf.ols("log_efficiency ~ age + C(sex) + C(help)", data=analysis_df).fit()

# Save summary to a text file for reference
with open("analysis_summary.txt", "w") as f:
    f.write("Linear model on efficiency (nuts/sec)\n")
    f.write(model.summary().as_text())
    f.write("\n\nLog-linear model on efficiency\n")
    f.write(log_model.summary().as_text())

# Print key results for quick check
print(model.summary())
print(log_model.summary())
