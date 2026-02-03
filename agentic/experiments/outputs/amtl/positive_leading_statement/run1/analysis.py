import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
csv_path = "amtl.csv"
df = pd.read_csv(csv_path)

# Basic cleaning: remove rows with missing or invalid sockets
# (sockets should be positive to model binomial proportions)
df = df.dropna(subset=["num_amtl", "sockets", "age", "prob_male", "tooth_class", "genus"]).copy()
df = df[df["sockets"] > 0].copy()

# Create binary indicator for modern humans
# Genus values include "Homo sapiens" for modern humans
df["is_human"] = (df["genus"] == "Homo sapiens").astype(int)

# Response as proportion with frequency weights = sockets
# Model: AMTL proportion ~ is_human + age + prob_male + tooth_class
# Use Binomial GLM
formula = "amtl_rate ~ is_human + age + prob_male + C(tooth_class)"

df["amtl_rate"] = df["num_amtl"] / df["sockets"]

model = smf.glm(
    formula=formula,
    data=df,
    family=sm.families.Binomial(),
    freq_weights=df["sockets"],
).fit()

# Save key results for inspection
with open("analysis_results.txt", "w") as f:
    f.write(model.summary().as_text())

# Also compute predicted difference between human and non-human at mean covariates
mean_age = df["age"].mean()
mean_prob_male = df["prob_male"].mean()

# Use most common tooth class for a reference prediction
ref_tooth_class = df["tooth_class"].mode().iloc[0]

predict_df = pd.DataFrame(
    {
        "is_human": [0, 1],
        "age": [mean_age, mean_age],
        "prob_male": [mean_prob_male, mean_prob_male],
        "tooth_class": [ref_tooth_class, ref_tooth_class],
    }
)

pred = model.get_prediction(predict_df).summary_frame()

with open("analysis_predictions.txt", "w") as f:
    f.write(pred.to_string(index=False))
