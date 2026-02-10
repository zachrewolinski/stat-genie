import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

# Load data
df = pd.read_csv("boxes.csv")

# Recode outcomes
# Social information use: 1 if child followed any demonstrated option (majority or minority), 0 if chose undemonstrated option
df["social_choice"] = np.where(df["y"] == 1, 0, 1)

# Majority preference: among demonstrated choices, 1 if majority, 0 if minority
demonstrated = df[df["y"].isin([2, 3])].copy()
demonstrated["majority_choice"] = np.where(demonstrated["y"] == 2, 1, 0)

# Age centering for stability
df["age_c"] = df["age"] - df["age"].mean()
demonstrated["age_c"] = demonstrated["age"] - demonstrated["age"].mean()

# Logistic regression models
# 1) Social information use as a function of age and culture
model_social = smf.logit("social_choice ~ age_c + C(culture)", data=df).fit(disp=False)

# 2) Majority vs minority choice as a function of age and culture
model_majority = smf.logit("majority_choice ~ age_c + C(culture)", data=demonstrated).fit(disp=False)

# Helper: predicted probabilities across cultures at mean age
def culture_variation(model, data, response_name):
    base = data.copy()
    base["age_c"] = 0.0
    # Unique cultures
    cultures = sorted(base["culture"].unique())
    probs = []
    for c in cultures:
        row = base.iloc[0:1].copy()
        row["culture"] = c
        p = float(model.predict(row)[0])
        probs.append(p)
    probs = np.array(probs)
    return cultures, probs

cultures_social, probs_social = culture_variation(model_social, df, "social_choice")
cultures_majority, probs_majority = culture_variation(model_majority, demonstrated, "majority_choice")

# Variation metrics across cultures (range of probabilities)
range_social = probs_social.max() - probs_social.min()
range_majority = probs_majority.max() - probs_majority.min()

# Age effects: predicted change across age span within cultures
def age_effect(model, data, culture_example):
    age_min = data["age_c"].min()
    age_max = data["age_c"].max()
    row_min = data.iloc[0:1].copy()
    row_max = data.iloc[0:1].copy()
    row_min["age_c"] = age_min
    row_max["age_c"] = age_max
    row_min["culture"] = culture_example
    row_max["culture"] = culture_example
    p_min = float(model.predict(row_min)[0])
    p_max = float(model.predict(row_max)[0])
    return abs(p_max - p_min)

# Use the most common culture as a representative for age effect
common_culture_social = int(df["culture"].value_counts().idxmax())
common_culture_majority = int(demonstrated["culture"].value_counts().idxmax())

age_effect_social = age_effect(model_social, df, common_culture_social)
age_effect_majority = age_effect(model_majority, demonstrated, common_culture_majority)

# Normalize metrics into [0,1] by capping at 1.0 (probabilities already bounded)
m1 = max(0.0, min(1.0, range_social))
m2 = max(0.0, min(1.0, range_majority))
m3 = max(0.0, min(1.0, age_effect_social))
m4 = max(0.0, min(1.0, age_effect_majority))

# Aggregate evidence: simple average of four metrics
evidence = (m1 + m2 + m3 + m4) / 4.0

# Map evidence in [0,1] to Likert scale [-100, 100]
scalar = int(round((2 * evidence - 1) * 100))

# Clip to bounds just in case
scalar = max(-100, min(100, scalar))

print("range_social", range_social)
print("range_majority", range_majority)
print("age_effect_social", age_effect_social)
print("age_effect_majority", age_effect_majority)
print("evidence", evidence)
print("scalar", scalar)

# Write scalar to conclusion.txt as required
with open("conclusion.txt", "w") as f:
    f.write(str(scalar))
