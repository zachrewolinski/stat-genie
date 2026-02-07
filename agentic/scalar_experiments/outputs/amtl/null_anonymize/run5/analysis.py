import pandas as pd
import numpy as np
import statsmodels.api as sm
import patsy

# Load data
csv_path = "amtl.csv"
df = pd.read_csv(csv_path)

# Rename for clarity
col_to_name = {
    "feature1": "tooth_class",
    "feature2": "specimen_id",
    "feature3": "missing_teeth",
    "feature4": "observable_sockets",
    "feature5": "age",
    "feature6": "age_uncertainty",
    "feature7": "sex",
    "feature8": "genus",
    "feature9": "region",
}

df = df.rename(columns=col_to_name)

# Basic cleaning
needed = ["tooth_class", "missing_teeth", "observable_sockets", "age", "sex", "genus"]
df = df.dropna(subset=needed).copy()

# Ensure numeric
for c in ["missing_teeth", "observable_sockets", "age", "sex"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")

df = df.dropna(subset=["missing_teeth", "observable_sockets", "age", "sex"]).copy()

# Remove invalid rows
# Observable sockets must be > 0 and missing <= observable
mask_valid = (df["observable_sockets"] > 0) & (df["missing_teeth"] >= 0) & (df["missing_teeth"] <= df["observable_sockets"])
df = df[mask_valid].copy()

# Create binary human indicator
df["is_human"] = (df["genus"] == "Homo sapiens").astype(int)

# Binomial counts: successes vs failures
X = patsy.dmatrix(
    "is_human + age + sex + C(tooth_class)",
    data=df,
    return_type="dataframe",
)
y = np.column_stack([df["missing_teeth"], df["observable_sockets"] - df["missing_teeth"]])

model = sm.GLM(y, X, family=sm.families.Binomial())
result = model.fit()

# Extract coefficient for is_human
coef = result.params.get("is_human", np.nan)
se = result.bse.get("is_human", np.nan)

z = coef / se if pd.notna(coef) and pd.notna(se) and se != 0 else 0.0

# Compute average predicted probability difference (human vs non-human)
# Use mean age and sex, and weight by observed tooth_class distribution
mean_age = df["age"].mean()
mean_sex = df["sex"].mean()

class_counts = df["tooth_class"].value_counts(normalize=True)

rows = []
for tc, w in class_counts.items():
    rows.append({"is_human": 1, "age": mean_age, "sex": mean_sex, "tooth_class": tc, "weight": w})
    rows.append({"is_human": 0, "age": mean_age, "sex": mean_sex, "tooth_class": tc, "weight": w})

pred_df = pd.DataFrame(rows)

pred_exog = patsy.dmatrix(
    "is_human + age + sex + C(tooth_class)",
    data=pred_df,
    return_type="dataframe",
)
pred_probs = result.predict(pred_exog)
pred_df["pred"] = pred_probs

human_pred = (pred_df[pred_df["is_human"] == 1]["pred"] * pred_df[pred_df["is_human"] == 1]["weight"]).sum()
nonhuman_pred = (pred_df[pred_df["is_human"] == 0]["pred"] * pred_df[pred_df["is_human"] == 0]["weight"]).sum()

pred_diff = human_pred - nonhuman_pred

# Map z-score to Likert-like scale
score = int(np.round(100 * np.tanh(z / 3.0)))

# Ensure sign aligns with predicted difference if available
if pd.notna(pred_diff) and pred_diff != 0:
    if pred_diff < 0 and score > 0:
        score = -score
    elif pred_diff > 0 and score < 0:
        score = -score

# Clamp to [-100, 100]
score = max(-100, min(100, score))

# Write conclusion
with open("conclusion.txt", "w", encoding="utf-8") as f:
    f.write(str(score))

# Optional: print key diagnostics for the user
print("is_human coef:", coef)
print("is_human z:", z)
print("predicted human rate:", human_pred)
print("predicted non-human rate:", nonhuman_pred)
print("predicted difference:", pred_diff)
print("score:", score)
