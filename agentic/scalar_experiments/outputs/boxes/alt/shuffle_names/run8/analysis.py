import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def main() -> None:
    cwd = Path(__file__).parent

    # Load metadata and data
    info = json.loads((cwd / "info.json").read_text())
    df = pd.read_csv(cwd / "boxes.csv")

    # Basic sanity checks
    expected_cols = {"majority_first", "gender", "age", "culture", "y"}
    if not expected_cols.issubset(df.columns):
        raise ValueError(f"Unexpected columns in data: {df.columns.tolist()}")

    # Derived variables
    df["choice_majority"] = (df["majority_first"] == 2).astype(int)
    df["choice_minority"] = (df["majority_first"] == 3).astype(int)
    df["choice_undemonstrated"] = (df["majority_first"] == 1).astype(int)

    # Reliance on social information: choosing any demonstrated option vs an undemonstrated third option
    df["choice_social"] = ((df["majority_first"] == 2) | (df["majority_first"] == 3)).astype(int)

    # Preference for majority cues: among social choices, majority vs minority
    df_social = df[df["choice_social"] == 1].copy()
    df_social["majority_over_minority"] = (df_social["majority_first"] == 2).astype(int)

    # Treat site ID as a categorical proxy for cultural context
    df["site"] = df["y"].astype("category")
    df_social["site"] = df_social["y"].astype("category")

    # Center age for numerical stability
    df["age_c"] = df["age"] - df["age"].mean()
    df_social["age_c"] = df_social["age"] - df_social["age"].mean()

    # Model 1: reliance on social information (any demonstrated option)
    # choice_social ~ age + site + culture (majority-first order)
    model_social = smf.logit("choice_social ~ age_c + C(site) + culture", data=df).fit(disp=False)

    # Model 2: majority preference among social choosers
    # majority_over_minority ~ age + site + culture
    model_majority = smf.logit(
        "majority_over_minority ~ age_c + C(site) + culture", data=df_social
    ).fit(disp=False)

    # Extract key statistics
    social_summary = model_social.summary2().tables[1]
    majority_summary = model_majority.summary2().tables[1]

    # Age effects
    age_effect_social = social_summary.loc["age_c"]
    age_effect_majority = majority_summary.loc["age_c"]

    # Site (culture) effects: any significant differences across sites?
    social_site_rows = social_summary.loc[[idx for idx in social_summary.index if idx.startswith("C(site)")]]
    majority_site_rows = majority_summary.loc[[idx for idx in majority_summary.index if idx.startswith("C(site)")]]

    any_social_site_sig = (social_site_rows["P>|z|"] < 0.05).any()
    any_majority_site_sig = (majority_site_rows["P>|z|"] < 0.05).any()

    # Overall descriptive patterns
    overall_social_rate = df["choice_social"].mean()
    overall_majority_rate = df["choice_majority"].mean()

    # Social choice by age (simple trend)
    age_social_trend = (
        df.groupby("age", observed=True)["choice_social"].mean().reset_index().sort_values("age")
    )
    age_majority_trend = (
        df.groupby("age", observed=True)["choice_majority"].mean().reset_index().sort_values("age")
    )

    # Summarize patterns in simple textual form
    def trend_direction(trend_df: pd.DataFrame, col: str) -> str:
        if len(trend_df) < 2:
            return "unclear"
        x = trend_df["age"].to_numpy()
        y_vals = trend_df[col].to_numpy()
        corr = np.corrcoef(x, y_vals)[0, 1]
        if corr > 0.2:
            return "increasing"
        if corr < -0.2:
            return "decreasing"
        return "flat"

    social_trend_dir = trend_direction(age_social_trend, "choice_social")
    majority_trend_dir = trend_direction(age_majority_trend, "choice_majority")

    # Map results to a Likert-style scalar answer (0–100)
    # We combine evidence from:
    # - Significance and sign of age effects (developmental change)
    # - Presence of significant site differences (cultural variation)
    # - Overall effect sizes / descriptive trends

    score = 50  # neutral baseline

    # Age effects on social reliance
    if age_effect_social["P>|z|"] < 0.05:
        # Positive coefficient => older children rely more on social information
        if age_effect_social["Coef."] > 0:
            score += 15
        else:
            score += 10  # developmental change but in the opposite direction

    # Age effects on majority preference
    if age_effect_majority["P>|z|"] < 0.05:
        if age_effect_majority["Coef."] > 0:
            score += 15
        else:
            score += 10

    # Cultural (site) differences
    if any_social_site_sig:
        score += 10
    if any_majority_site_sig:
        score += 10

    # Descriptive trend nudges
    if social_trend_dir == "increasing":
        score += 5
    elif social_trend_dir == "decreasing":
        score += 0

    if majority_trend_dir == "increasing":
        score += 5
    elif majority_trend_dir == "decreasing":
        score += 0

    # Clamp to [0, 100]
    score = int(max(0, min(100, round(score))))

    # Build explanation text
    rq = info["research_questions"][0]

    explanation_lines = []
    explanation_lines.append(
        f"Research question: {rq}"
    )
    explanation_lines.append(
        "I analyzed 629 trials where children (ages 4–14) chose between a majority-demonstrated option, "
        "a minority-demonstrated option, or an undemonstrated third option."
    )
    explanation_lines.append(
        "Reliance on social information was defined as choosing any demonstrated option (majority or minority) "
        "versus the undemonstrated option."
    )
    explanation_lines.append(
        "Preference for majority cues was defined, among socially guided choices, as choosing the majority "
        "rather than the minority option."
    )

    explanation_lines.append(
        f"Overall, children relied on social information on roughly {overall_social_rate:.0%} of trials "
        f"and chose the majority option on about {overall_majority_rate:.0%} of trials."
    )

    explanation_lines.append(
        "A logistic regression of social versus nonsocial choices on age, site (as a categorical proxy for cultural "
        "context), and the majority-first order variable showed the following for age: "
        f"coefficient={age_effect_social['Coef.']:.3f}, p-value={age_effect_social['P>|z|']:.3f}."
    )
    explanation_lines.append(
        "This indicates that reliance on social information "
        f"{'increases' if age_effect_social['Coef.'] > 0 else 'decreases'} with age "
        f"and that this developmental trend is "
        f"{'statistically reliable (p<0.05)' if age_effect_social['P>|z|'] < 0.05 else 'not statistically reliable (p≥0.05)'}."
    )

    explanation_lines.append(
        "For majority preference among children who followed social information, a logistic regression of "
        "majority- versus minority-choices on age, site, and majority-first order yielded "
        f"age coefficient={age_effect_majority['Coef.']:.3f}, p-value={age_effect_majority['P>|z|']:.3f}."
    )
    explanation_lines.append(
        "This suggests that majority preference "
        f"{'increases' if age_effect_majority['Coef.'] > 0 else 'decreases'} with age and that this age effect is "
        f"{'statistically reliable (p<0.05)' if age_effect_majority['P>|z|'] < 0.05 else 'not statistically reliable (p≥0.05)'}."
    )

    explanation_lines.append(
        "Including site as a categorical predictor revealed "
        f"{'significant' if any_social_site_sig or any_majority_site_sig else 'little clear'} variation across sites: "
        f"site terms reached p<0.05 for social-reliance={'yes' if any_social_site_sig else 'no'}, "
        f"and for majority-preference={'yes' if any_majority_site_sig else 'no'}."
    )

    explanation_lines.append(
        f"Descriptively, the proportion of social choices across ages showed an {social_trend_dir} trend, "
        f"while the proportion of majority choices showed an {majority_trend_dir} trend."
    )

    if score >= 67:
        overall_answer = (
            "Taken together, these results provide substantial evidence that both children's reliance on social "
            "information and their preference for majority cues vary systematically with age and across cultural "
            "sites in this dataset."
        )
    elif score <= 33:
        overall_answer = (
            "Taken together, the evidence for systematic variation in children's reliance on social information and "
            "preference for majority cues across age and cultural sites is weak or inconclusive in this dataset."
        )
    else:
        overall_answer = (
            "Taken together, the dataset provides mixed evidence that children's reliance on social information and "
            "preference for majority cues vary with age and across cultural sites: some effects are suggestive but not "
            "uniformly strong or statistically robust."
        )

    explanation_lines.append(overall_answer)
    explanation_lines.append(
        f"On a 0–100 scale where higher values represent a stronger 'Yes' answer to the research question, "
        f"I assign a response value of {score}, reflecting the combined strength of the observed developmental and "
        f"cross-cultural patterns."
    )

    explanation = " ".join(explanation_lines)

    conclusion = {"response": score, "explanation": explanation}
    (cwd / "conclusion.txt").write_text(json.dumps(conclusion))


if __name__ == "__main__":
    main()

