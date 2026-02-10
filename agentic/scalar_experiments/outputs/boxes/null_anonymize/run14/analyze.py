import json
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).parent

info = json.loads((BASE / "info.json").read_text())

question = info["research_questions"][0]

print("Research question:\n", question, "\n", sep="")

cols = info["data_desc"]["field_names"]

print("Columns:", cols)

df = pd.read_csv(BASE / "boxes.csv")
print("\nShape:", df.shape)

print("\nOutcome (feature1) value counts:")
print(df["feature1"].value_counts().sort_index())

print("\nOutcome proportions:")
print(df["feature1"].value_counts(normalize=True).sort_index())

print("\nAge (feature3) summary:")
print(df["feature3"].describe())

print("\nOutcome by age quartile:")
age_bins = pd.qcut(df["feature3"], 4, labels=["Q1_youngest", "Q2", "Q3", "Q4_oldest"])
print(pd.crosstab(age_bins, df["feature1"], normalize="index"))

print("\nOutcome by site (feature5) proportions:")
print(pd.crosstab(df["feature5"], df["feature1"], normalize="index"))

print("\nMean majority-choice indicator (feature1==2):", (df["feature1"] == 2).mean())

coef = (df["feature1"] == 2).mean()

age = df["feature3"]
from statsmodels.formula.api import logit

df2 = df.copy()
df2["majority"] = (df2["feature1"] == 2).astype(int)

model = logit("majority ~ feature3 + C(feature5)", data=df2).fit(disp=False)
print("\nLogit summary (majority ~ age + site):")
print(model.summary())

age_effect = model.params.get("feature3", 0.0)
print("\nAge coefficient for majority choice:", age_effect)

# crude scalar mapping: center around neutral 0.5 majority
strength = (coef - 0.5) * 200
# also incorporate age trend sign
if age_effect > 0:
    strength += min(20, age_effect * 100)
elif age_effect < 0:
    strength -= min(20, abs(age_effect) * 100)

strength = int(max(-100, min(100, round(strength))))

print("\nDerived Likert scalar:", strength)

(Path("conclusion.txt")).write_text(str(strength))
