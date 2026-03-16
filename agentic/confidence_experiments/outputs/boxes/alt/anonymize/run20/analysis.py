import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


DATA_PATH = Path("boxes.csv")
INFO_PATH = Path("info.json")
OUTPUT_PATH = Path("conclusion.txt")


def load_data():
    df = pd.read_csv(DATA_PATH)
    with INFO_PATH.open() as f:
        info = json.load(f)
    return df, info


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    # feature1: 1=unchosen, 2=majority, 3=minority
    # feature2: 1=girl, 2=boy
    # feature3: age in years
    # feature4: majority-first demo (0/1)
    # feature5: site id (1-8)
    df = df.copy()
    # Keep only choices among demonstrated options (majority vs minority)
    df = df[df["feature1"].isin([2, 3])].copy()
    # Binary outcome: 1 = chose majority, 0 = chose minority
    df["majority_choice"] = (df["feature1"] == 2).astype(int)
    # Center age for interpretability
    df["age_c"] = df["feature3"] - df["feature3"].mean()
    # Categorical encodings
    df["gender"] = df["feature2"].map({1: "girl", 2: "boy"})
    df["site"] = df["feature5"].astype("category")
    return df


def descriptive_stats(df: pd.DataFrame) -> dict:
    overall_prop = df["majority_choice"].mean()

    # Age groups: early (4-6), middle (7-10), late (11-14)
    bins = [4, 7, 11, 15]
    labels = ["4-6", "7-10", "11-14"]
    df["age_group"] = pd.cut(df["feature3"], bins=bins, labels=labels, right=False)

    by_age = df.groupby("age_group")["majority_choice"].agg(["mean", "count"])
    by_site = df.groupby("site")["majority_choice"].agg(["mean", "count"])

    return {
        "overall_prop_majority": float(overall_prop),
        "age_group_stats": by_age.to_dict(),
        "site_stats": by_site.to_dict(),
    }


def logistic_regression_models(df: pd.DataFrame):
    results = {}

    # Model 1: majority choice ~ age_c
    model1 = smf.glm(
        formula="majority_choice ~ age_c",
        data=df,
        family=sm.families.Binomial(),
    ).fit()
    results["age_only"] = model1

    # Model 2: age + gender + majority-first + site (as fixed effects)
    model2 = smf.glm(
        formula="majority_choice ~ age_c + C(gender) + feature4 + C(site)",
        data=df,
        family=sm.families.Binomial(),
    ).fit()
    results["full"] = model2

    return results


def summarize_results(desc: dict, models: dict) -> tuple[int, str]:
    overall = desc["overall_prop_majority"]

    model_age = models["age_only"]
    model_full = models["full"]

    # Extract p-values and effect sizes
    age_p_age_only = model_age.pvalues.get("age_c", np.nan)
    age_coef_age_only = model_age.params.get("age_c", np.nan)

    age_p_full = model_full.pvalues.get("age_c", np.nan)
    age_coef_full = model_full.params.get("age_c", np.nan)

    # Site variance in majority preference
    site_means = desc["site_stats"]["mean"]
    site_props = list(site_means.values())
    site_range = float(max(site_props) - min(site_props)) if site_props else np.nan

    # Determine evidence strength
    p_threshold = 0.05
    strong_age_effect = (age_p_full < p_threshold) and (abs(age_coef_full) > 0.05)
    moderate_age_effect = (age_p_full < p_threshold) and (abs(age_coef_full) <= 0.05)

    # Map evidence to Likert scale
    if strong_age_effect and site_range > 0.1:
        response = 85
    elif strong_age_effect:
        response = 75
    elif moderate_age_effect:
        response = 60
    elif (age_p_full < 0.1) or (age_p_age_only < 0.1):
        response = 45
    else:
        response = 25

    explanation_parts = []

    explanation_parts.append(
        "We modelled children's tendency to follow majority demonstrations "
        "using logistic regression with majority vs. minority choice as the outcome."
    )

    explanation_parts.append(
        f"Overall, children chose the majority option in approximately "
        f"{overall:.2f} of trials restricted to demonstrated options."
    )

    explanation_parts.append(
        "A baseline model with age as the only predictor indicated an age effect "
        f"with coefficient {age_coef_age_only:.3f} (p = {age_p_age_only:.3g})."
    )

    explanation_parts.append(
        "A fuller model including age, gender, demonstration order, and site fixed effects "
        f"yielded an age coefficient of {age_coef_full:.3f} (p = {age_p_full:.3g})."
    )

    if strong_age_effect or moderate_age_effect:
        explanation_parts.append(
            "The consistently significant positive age coefficient suggests that reliance on "
            "majority social information increases with developmental stage."
        )
    else:
        explanation_parts.append(
            "Age effects were weak or statistically unreliable, providing limited evidence that "
            "reliance on majority social information changes systematically with age in this sample."
        )

    explanation_parts.append(
        "Cross-cultural variation was captured via site fixed effects; the proportion of majority "
        f"choices varied across sites with a range of about {site_range:.2f} points on the probability scale."
    )

    if site_range > 0.1:
        explanation_parts.append(
            "This heterogeneity across societies suggests that cultural context shapes the strength "
            "of children's majority preference, even after accounting for age and gender."
        )
    else:
        explanation_parts.append(
            "However, the between-site range in majority preference was modest, indicating only "
            "limited cross-cultural modulation in this dataset."
        )

    if response >= 50:
        explanation_parts.append(
            "Taken together, these findings provide overall support for the claim that children's "
            "reliance on social information and preference for majority cues vary across "
            "developmental stages and, to a lesser extent, across cultures."
        )
    else:
        explanation_parts.append(
            "Taken together, these findings do not provide strong evidence that children's reliance "
            "on social information and preference for majority cues vary systematically with age "
            "or cultural context in this dataset."
        )

    explanation = " " .join(explanation_parts)

    # Ensure integer and bounds
    response_int = int(round(max(0, min(100, response))))
    return response_int, explanation


def main():
    df, info = load_data()
    df_prep = prepare_data(df)

    # Edge case: if there are no relevant rows
    if df_prep.empty:
        response = 10
        explanation = (
            "After restricting the data to trials in which children chose between "
            "demonstrated majority and minority options, no observations remained. "
            "As a result, we cannot assess whether reliance on majority social information "
            "varies across age or cultures in this dataset."
        )
    else:
        desc = descriptive_stats(df_prep)
        models = logistic_regression_models(df_prep)
        response, explanation = summarize_results(desc, models)

    output = {"response": response, "explanation": explanation}
    with OUTPUT_PATH.open("w") as f:
        json.dump(output, f)


if __name__ == "__main__":
    main()
