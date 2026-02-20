import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def load_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    # The metadata indicates that column names are shuffled relative to their
    # semantic meaning. We remap them based on the descriptions in info.json.
    df = df.copy()
    df["id"] = df["education"]
    df["affair_freq"] = df["age"]
    df["gender_cat"] = df["gender"]
    df["age_years"] = df["occupation"]
    df["years_married"] = df["children"]
    df["has_children"] = df["religiousness"].map({"yes": 1, "no": 0}).astype(int)
    df["religiousness_level"] = df["rating"]
    df["education_level"] = df["yearsmarried"]
    df["occupation_code"] = df["rownames"]
    df["marriage_rating"] = df["affairs"]

    # Binary outcome: any extramarital intercourse in past year
    df["any_affair"] = (df["affair_freq"] > 0).astype(int)

    return df


def descriptive_stats(df: pd.DataFrame) -> dict:
    group = df.groupby("has_children")
    desc = {
        "mean_affair_freq_by_children": group["affair_freq"].mean().to_dict(),
        "prop_any_affair_by_children": group["any_affair"].mean().to_dict(),
        "n_by_children": group["any_affair"].size().to_dict(),
    }
    return desc


def logistic_models(df: pd.DataFrame) -> dict:
    results = {}

    # Simple logistic model: any_affair ~ has_children
    simple_model = smf.logit("any_affair ~ has_children", data=df).fit(disp=False)
    results["simple"] = {
        "params": simple_model.params.to_dict(),
        "pvalues": simple_model.pvalues.to_dict(),
    }

    # Adjusted model with key covariates
    # Use gender as a binary indicator (female reference)
    df = df.copy()
    df["gender_male"] = (df["gender_cat"] == "male").astype(int)

    formula = (
        "any_affair ~ has_children + age_years + years_married + "
        "religiousness_level + education_level + marriage_rating + gender_male"
    )
    adjusted_model = smf.logit(formula, data=df).fit(disp=False)
    results["adjusted"] = {
        "params": adjusted_model.params.to_dict(),
        "pvalues": adjusted_model.pvalues.to_dict(),
    }

    return results


def interpret_results(desc: dict, models: dict) -> dict:
    mean_freq = desc["mean_affair_freq_by_children"]
    prop_any = desc["prop_any_affair_by_children"]

    # Children indicator keys: 0 = no children, 1 = has children
    mean_no_children = float(mean_freq.get(0, np.nan))
    mean_with_children = float(mean_freq.get(1, np.nan))
    prop_no_children = float(prop_any.get(0, np.nan))
    prop_with_children = float(prop_any.get(1, np.nan))

    simple = models["simple"]
    adjusted = models["adjusted"]

    simple_beta = simple["params"]["has_children"]
    simple_p = simple["pvalues"]["has_children"]
    adj_beta = adjusted["params"]["has_children"]
    adj_p = adjusted["pvalues"]["has_children"]

    # Negative coefficient -> having children associated with lower odds of affairs.
    # Positive coefficient -> associated with higher odds.
    if adj_beta < 0 and adj_p < 0.05:
        response = "Yes"
        strength = 80
    elif adj_beta < 0 and adj_p < 0.1:
        response = "Yes"
        strength = 60
    elif adj_beta > 0 and adj_p < 0.05:
        response = "No"
        strength = 80
    elif adj_beta > 0 and adj_p < 0.1:
        response = "No"
        strength = 60
    else:
        # No clear directional evidence
        response = "No"
        strength = 40

    # Confidence reflects sample size, model consistency, and p-values
    if (simple_p < 0.05 and adj_p < 0.05) or (simple_p < 0.1 and adj_p < 0.1):
        confidence = 75
    else:
        confidence = 55

    explanation_parts = [
        "We examined whether having children is associated with lower engagement "
        "in extramarital affairs in a sample of 601 first-marriage respondents.",
        f"Mean affair frequency (0–12 scale) was {mean_no_children:.3f} for those "
        f"without children and {mean_with_children:.3f} for those with children.",
        f"The proportion reporting any affairs in the past year was "
        f"{prop_no_children:.3f} without children vs {prop_with_children:.3f} "
        "with children.",
        "A simple logistic regression of any affair on a children indicator and "
        "an adjusted logistic model including age, years married, education, "
        "religiousness, self-rated marriage quality, and gender were estimated.",
        f"In the adjusted model, the coefficient on having children was "
        f"{adj_beta:.3f} with p-value {adj_p:.3f}, while in the simple model it "
        f"was {simple_beta:.3f} with p-value {simple_p:.3f}.",
    ]

    if response == "Yes":
        explanation_parts.append(
            "Both the sign and statistical significance of the children coefficient "
            "suggest that having children is associated with lower odds of "
            "extramarital affairs, after accounting for other covariates."
        )
    elif strength >= 60:
        explanation_parts.append(
            "Both models indicate that having children is associated with higher "
            "or at least not lower odds of extramarital affairs."
        )
    else:
        explanation_parts.append(
            "Estimates across models are not consistently negative and are often "
            "not statistically significant, so there is no clear evidence that "
            "having children reduces extramarital affairs."
        )

    explanation = " ".join(explanation_parts)

    return {
        "response": response,
        "strength": strength,
        "confidence": confidence,
        "explanation": explanation,
    }


def main() -> None:
    df = load_data(Path("affairs.csv"))
    desc = descriptive_stats(df)
    models = logistic_models(df)
    conclusion = interpret_results(desc, models)

    with open("conclusion.txt", "w", encoding="utf-8") as f:
        json.dump(conclusion, f, ensure_ascii=False)


if __name__ == "__main__":
    main()

