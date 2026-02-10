import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def main() -> None:
    root = Path(__file__).parent
    info_path = root / "info.json"
    data_path = root / "boxes.csv"

    with info_path.open("r", encoding="utf-8") as f:
        info = json.load(f)

    df = pd.read_csv(data_path)

    # Create key outcome measures
    df["majority_choice"] = (df["y"] == 2).astype(int)
    df["demonstrated_choice"] = df["y"].isin([2, 3]).astype(int)

    # Center age to improve model stability
    df["age_c"] = df["age"] - df["age"].mean()

    # Logistic regression: preference for majority cues
    # majority_choice ~ age + gender + majority_first + culture
    model_formula = "majority_choice ~ age_c + gender + majority_first + C(culture)"
    logit_model = smf.logit(model_formula, data=df).fit(disp=False)
    summary = logit_model.summary2().tables[1]

    # Extract evidence for developmental and cultural variation
    # Age effect
    age_p = float(summary.loc["age_c", "P>|z|"])
    age_coef = float(summary.loc["age_c", "Coef."])

    # Culture effects: any culture dummy with p < 0.05 and non-trivial magnitude
    culture_rows = summary.loc[[idx for idx in summary.index if idx.startswith("C(culture)[T.")]]
    culture_significant = (culture_rows["P>|z|"] < 0.05).any()

    # Simple descriptive checks (by age and culture)
    majority_by_age = (
        df.groupby("age", as_index=False)["majority_choice"].mean().rename(columns={"majority_choice": "rate"})
    )
    age_range_change = float(majority_by_age["rate"].max() - majority_by_age["rate"].min())

    majority_by_culture = (
        df.groupby("culture", as_index=False)["majority_choice"].mean().rename(columns={"majority_choice": "rate"})
    )
    culture_range_change = float(majority_by_culture["rate"].max() - majority_by_culture["rate"].min())

    # Heuristic mapping from evidence to Likert-scale conclusion
    # We are answering: "Do children’s reliance on social information and preference
    # for majority cues vary across cultures and developmental stages?"
    evidence_score = 0.0

    # Age evidence: statistical significance and effect direction
    if age_p < 0.001:
        evidence_score += 40.0 * np.sign(age_coef)
    elif age_p < 0.01:
        evidence_score += 30.0 * np.sign(age_coef)
    elif age_p < 0.05:
        evidence_score += 20.0 * np.sign(age_coef)

    # Age descriptive range (difference in majority-following rates across ages)
    if age_range_change > 0.25:
        evidence_score += 25.0
    elif age_range_change > 0.15:
        evidence_score += 15.0
    elif age_range_change > 0.05:
        evidence_score += 5.0

    # Culture evidence: any significant culture coefficients and descriptive spread
    if culture_significant:
        evidence_score += 25.0

    if culture_range_change > 0.25:
        evidence_score += 20.0
    elif culture_range_change > 0.15:
        evidence_score += 10.0
    elif culture_range_change > 0.05:
        evidence_score += 5.0

    # Clamp to Likert scale [-100, 100]
    scalar = int(np.clip(np.round(evidence_score), -100, 100))

    # If there is essentially no evidence of variation, pull score toward "No"
    if scalar == 0 and age_p >= 0.1 and not culture_significant and age_range_change < 0.05 and culture_range_change < 0.05:
        scalar = -20

    # Write final scalar conclusion only
    conclusion_path = root / "conclusion.txt"
    with conclusion_path.open("w", encoding="utf-8") as f:
        f.write(str(scalar))


if __name__ == "__main__":
    main()

