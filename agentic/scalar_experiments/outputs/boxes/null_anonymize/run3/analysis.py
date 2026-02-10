import json
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent

info_path = BASE_DIR / "info.json"
boxes_path = BASE_DIR / "boxes.csv"

with info_path.open() as f:
    info = json.load(f)

research_q = info["research_questions"][0]

# Load data
df = pd.read_csv(boxes_path)

# Rename columns for clarity
df = df.rename(
    columns={
        "feature1": "choice",          # 1=undemonstrated, 2=majority, 3=minority
        "feature2": "gender",          # 1=girl, 2=boy
        "feature3": "age",             # age in years
        "feature4": "majority_first",  # 0/1
        "feature5": "site",            # cultural site ID
    }
)

# Basic derived measures
n = len(df)
majority_share = (df["choice"] == 2).mean()
minority_share = (df["choice"] == 3).mean()
undemo_share = (df["choice"] == 1).mean()

# Age groups (rough developmental stages)
bins = [3.5, 6.5, 9.5, 11.5, 14.5]
labels = ["4-6", "7-9", "10-11", "12-14"]
df["age_group"] = pd.cut(df["age"], bins=bins, labels=labels)

age_group_majority = df.groupby("age_group")["choice"].apply(lambda s: (s == 2).mean())

# Site-level majority preference
site_majority = df.groupby("site")["choice"].apply(lambda s: (s == 2).mean())

# Quantify variation across cultures (sites) and age groups
site_var = site_majority.var(ddof=1) if len(site_majority) > 1 else 0.0
age_var = age_group_majority.var(ddof=1) if len(age_group_majority) > 1 else 0.0

# Overall reliance on social information and majority cues.
# Here, "reliance on social information" is operationalized as
# the proportion of choices that follow either majority or minority
# demonstrations vs. undemonstrated option.
social_reliance = 1.0 - undemo_share

# Preference for majority over minority when using social info.
if social_reliance > 0:
    majority_given_social = majority_share / social_reliance
else:
    majority_given_social = 0.0

# Build a scalar in [-100, 100].
# Positive values indicate stronger and more consistent majority preference
# across cultures and ages; negatives would indicate absence or reversal.
# Components:
#   - overall majority_given_social (0-1, center 0.5)
#   - penalize high cross-site and cross-age variance

strength_component = (majority_given_social - 0.5) * 2  # maps 0-1 -> [-1,1] centered at 0.5

# Normalize variance penalties using rough scale factors so typical var ~0.02-0.05
# does not swamp the signal but reduces extreme heterogeneity.
var_penalty = 0.0
var_penalty -= min(site_var / 0.05, 1.0) * 0.3
var_penalty -= min(age_var / 0.02, 1.0) * 0.3

raw_score = strength_component + var_penalty

# Clip to [-1,1] and scale to [-100,100]
raw_score = max(min(raw_score, 1.0), -1.0)
likert_score = int(round(raw_score * 100))

# Ensure within bounds
likert_score = max(min(likert_score, 100), -100)

conclusion_path = BASE_DIR / "conclusion.txt"
conclusion_path.write_text(str(likert_score))
