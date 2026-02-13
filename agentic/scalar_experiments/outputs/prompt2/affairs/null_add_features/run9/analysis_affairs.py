import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def load_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # Ensure expected columns are present
    required_cols = {
        "affairs",
        "children",
        "age",
        "yearsmarried",
        "religiousness",
        "education",
        "occupation",
        "rating",
        "gender",
    }
    missing = required_cols.difference(df.columns)
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")

    return df


def compute_descriptives(df: pd.DataFrame) -> dict:
    df = df.copy()
    df["has_affair"] = (df["affairs"] > 0).astype(int)

    group = df.groupby("children", observed=True)

    descriptives = {
        "mean_affairs_by_children": group["affairs"].mean().to_dict(),
        "prop_any_affair_by_children": group["has_affair"].mean().to_dict(),
        "n_by_children": group.size().to_dict(),
    }
    return descriptives


def fit_models(df: pd.DataFrame) -> dict:
    df = df.copy()
    df["has_affair"] = (df["affairs"] > 0).astype(int)

    # Logistic regression for probability of any affair
    logit_formula = (
        "has_affair ~ C(children) + age + yearsmarried + "
        "religiousness + education + occupation + rating + C(gender)"
    )
    logit_model = smf.logit(logit_formula, data=df).fit(disp=False)

    # Poisson regression for affair count (acknowledging coarse coding)
    poisson_formula = (
        "affairs ~ C(children) + age + yearsmarried + "
        "religiousness + education + occupation + rating + C(gender)"
    )
    poisson_model = smf.glm(
        poisson_formula, data=df, family=sm.families.Poisson()
    ).fit()

    return {"logit": logit_model, "poisson": poisson_model}


def interpret_effect(
    descriptives: dict, models: dict
) -> tuple[str, int, str]:
    mean_affairs = descriptives["mean_affairs_by_children"]
    prop_any = descriptives["prop_any_affair_by_children"]

    # Normalize keys to “yes” / “no” strings if needed
    def get_key(d, key):
        # Handle possible capitalization differences
        for k in d.keys():
            if str(k).lower() == key:
                return d[k]
        return np.nan

    mean_yes = get_key(mean_affairs, "yes")
    mean_no = get_key(mean_affairs, "no")
    prop_yes = get_key(prop_any, "yes")
    prop_no = get_key(prop_any, "no")

    logit = models["logit"]
    poisson = models["poisson"]

    # In statsmodels coding, children=no is baseline; coefficient is for children[T.yes]
    logit_params = logit.params
    poisson_params = poisson.params
    logit_pvalues = logit.pvalues
    poisson_pvalues = poisson.pvalues

    coef_logit_children = logit_params.get("C(children)[T.yes]", np.nan)
    p_logit_children = logit_pvalues.get("C(children)[T.yes]", np.nan)

    coef_pois_children = poisson_params.get("C(children)[T.yes]", np.nan)
    p_pois_children = poisson_pvalues.get("C(children)[T.yes]", np.nan)

    # Basic direction based on raw descriptives
    direction_descriptive = None
    if not np.isnan(mean_yes) and not np.isnan(mean_no):
        if mean_yes < mean_no:
            direction_descriptive = "decrease"
        elif mean_yes > mean_no:
            direction_descriptive = "increase"
        else:
            direction_descriptive = "no_difference"

    # Determine statistical evidence (using 0.05 threshold)
    evidence_decrease = (
        coef_logit_children < 0
        and p_logit_children < 0.05
        and coef_pois_children < 0
        and p_pois_children < 0.05
    )

    evidence_increase = (
        coef_logit_children > 0
        and p_logit_children < 0.05
        and coef_pois_children > 0
        and p_pois_children < 0.05
    )

    if evidence_decrease and direction_descriptive == "decrease":
        response = "Yes"
        base_conf = 85
    elif evidence_increase and direction_descriptive == "increase":
        response = "No"
        base_conf = 85
    else:
        # No consistent evidence that children reduce affairs
        response = "No"
        base_conf = 70

    # Adjust confidence modestly based on strength of evidence
    if response == "Yes" and evidence_decrease and p_logit_children < 0.01:
        base_conf = 90
    elif response == "No" and evidence_increase and p_logit_children < 0.01:
        base_conf = 90

    # Build explanation
    explanation_parts = []
    explanation_parts.append(
        "I analyzed a sample of 601 married individuals from the Fair affairs dataset, "
        "examining whether having children is associated with lower engagement in extramarital affairs."
    )

    if not np.isnan(mean_yes) and not np.isnan(mean_no):
        explanation_parts.append(
            f"On average, individuals with children reported {mean_yes:.3f} affair-score units "
            f"versus {mean_no:.3f} for those without children."
        )
    if not np.isnan(prop_yes) and not np.isnan(prop_no):
        explanation_parts.append(
            f"The proportion reporting any extramarital activity (affairs > 0) was "
            f"{prop_yes:.3f} among those with children and {prop_no:.3f} among those without."
        )

    explanation_parts.append(
        "To adjust for potential confounders (age, years married, religiousness, education, "
        "occupation, self-rated marriage quality, and gender), I fit two regression models: "
        "a logistic model for the probability of any affair and a Poisson model for the coded affair count."
    )

    if not np.isnan(coef_logit_children):
        explanation_parts.append(
            "In the logistic model, the coefficient for having children "
            f"(relative to no children) was {coef_logit_children:.3f} with p-value {p_logit_children:.3f}."
        )
    if not np.isnan(coef_pois_children):
        explanation_parts.append(
            "In the Poisson model, the coefficient for having children was "
            f"{coef_pois_children:.3f} with p-value {p_pois_children:.3f}."
        )

    if response == "Yes":
        explanation_parts.append(
            "Both the descriptive statistics and the regression models indicate that, "
            "after accounting for other variables, having children is associated with a lower "
            "level of extramarital-affair engagement."
        )
    else:
        explanation_parts.append(
            "Taken together, the descriptive differences and regression results do not provide "
            "strong, consistent evidence that having children reduces extramarital-affair engagement; "
            "any observed differences are small or not statistically robust."
        )

    explanation = " ".join(explanation_parts)

    confidence = int(round(base_conf))
    confidence = max(0, min(100, confidence))

    return response, confidence, explanation


def main() -> None:
    csv_path = Path("affairs.csv")
    df = load_data(csv_path)

    descriptives = compute_descriptives(df)
    models = fit_models(df)
    response, confidence, explanation = interpret_effect(descriptives, models)

    conclusion = {
        "response": response,
        "confidence": confidence,
        "explanation": explanation,
    }

    conclusion_path = Path("conclusion.txt")
    with conclusion_path.open("w", encoding="utf-8") as f:
        json.dump(conclusion, f, ensure_ascii=False)


if __name__ == "__main__":
    main()

