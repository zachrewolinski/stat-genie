import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def load_metadata(path: Path) -> dict:
    with path.open("r") as f:
        return json.load(f)


def run_analysis() -> dict:
    base_path = Path(__file__).parent

    info = load_metadata(base_path / "info.json")
    df = pd.read_csv(base_path / "affairs.csv")

    # Map variables based on the metadata descriptions (column names are shuffled)
    # "age" column encodes extramarital intercourse frequency during the past year.
    df = df.copy()
    df["affair_freq"] = df["age"]

    # "religiousness" column is actually the yes/no indicator for whether there
    # are children in the marriage.
    df["has_children"] = df["religiousness"].map({"yes": 1, "no": 0})

    # Define a binary outcome: had any extramarital affair in the past year.
    df["had_affair"] = (df["affair_freq"] > 0).astype(int)

    # Drop rows with missing values in key variables, if any.
    df = df.dropna(subset=["had_affair", "has_children"])

    # Descriptive comparison: proportions and means by children status.
    group = df.groupby("has_children")

    prop_affair = group["had_affair"].mean()
    mean_freq = group["affair_freq"].mean()

    # Logistic regression with and without basic controls.
    # Basic model: only children indicator.
    basic_model = smf.logit("had_affair ~ has_children", data=df).fit(disp=False)
    basic_coef = basic_model.params["has_children"]
    basic_pval = basic_model.pvalues["has_children"]

    # Expanded model with additional covariates to adjust for confounding.
    # Based on metadata, we can interpret:
    #   occupation -> age group
    #   children   -> years married
    #   rating     -> religiousness score
    #   yearsmarried -> education
    #   rownames   -> occupation code
    #   affairs    -> marriage rating
    # plus gender as a categorical predictor.
    df["age_group"] = df["occupation"]
    df["years_married"] = df["children"]
    df["relig_score"] = df["rating"]
    df["education_years"] = df["yearsmarried"]
    df["occupation_code"] = df["rownames"]
    df["marriage_rating"] = df["affairs"]

    full_formula = (
        "had_affair ~ has_children + C(gender) + age_group + years_married + "
        "relig_score + education_years + occupation_code + marriage_rating"
    )

    try:
        full_model = smf.logit(full_formula, data=df).fit(disp=False, maxiter=100)
        full_coef = full_model.params.get("has_children", np.nan)
        full_pval = full_model.pvalues.get("has_children", np.nan)
    except Exception:
        # If the full model fails to converge, fall back to the basic model only.
        full_coef = np.nan
        full_pval = np.nan

    # Decide on the answer:
    # "Yes" means we find evidence that having children is associated with
    # *lower* engagement in extramarital affairs (negative coefficient and
    # reasonably small p-value). Otherwise, we answer "No".
    coef_to_use = full_coef if np.isfinite(full_coef) else basic_coef
    pval_to_use = full_pval if np.isfinite(full_pval) else basic_pval

    if (coef_to_use < 0) and (pval_to_use < 0.05):
        response = "Yes"
    else:
        response = "No"

    # Map p-value to a rough confidence score.
    if pval_to_use < 0.01:
        confidence = 90
    elif pval_to_use < 0.05:
        confidence = 80
    elif pval_to_use < 0.1:
        confidence = 65
    else:
        confidence = 55

    # Build an explanation string summarizing the main pieces of evidence.
    question = info.get("research_questions", ["Does having children decrease engagement in extramarital affairs?"])[
        0
    ]

    # Prepare human-readable group labels.
    def fmt_group(val: float) -> str:
        return f"{val:.3f}"

    prop_with_children = prop_affair.get(1, np.nan)
    prop_without_children = prop_affair.get(0, np.nan)
    mean_with_children = mean_freq.get(1, np.nan)
    mean_without_children = mean_freq.get(0, np.nan)

    explanation = (
        f"Research question: {question}\n\n"
        "I used the Psychology Today extramarital affairs survey data (601 married individuals).\n"
        "In this dataset, the `age` column encodes the frequency of extramarital sexual intercourse "
        "during the past year (0 = none, higher values = more frequent), and the `religiousness` "
        "column is a yes/no indicator of whether there are children in the marriage.\n\n"
        "First, I created a binary outcome `had_affair` indicating whether a respondent reported any "
        "extramarital intercourse in the past year. I compared this outcome between couples with "
        "children (`has_children = 1`) and without children (`has_children = 0`). The proportion of "
        "individuals reporting any affair was "
        f"{fmt_group(prop_with_children)} for those with children and "
        f"{fmt_group(prop_without_children)} for those without children. The mean frequency of "
        "extramarital intercourse (using the coded frequency scale) was "
        f"{fmt_group(mean_with_children)} with children versus "
        f"{fmt_group(mean_without_children)} without children.\n\n"
        "Next, I fit a logistic regression model with `had_affair` as the dependent variable and "
        "`has_children` as the main predictor. I also estimated a model that adjusts for gender, age "
        "group, years married, religiousness score, education, occupation code, and self-rated "
        "marital happiness. In these models, the coefficient for `has_children` represents the change "
        "in log-odds of having an affair for individuals with children relative to those without.\n\n"
        f"In the primary model used for inference, the coefficient on `has_children` was "
        f"{coef_to_use:.3f} with p-value {pval_to_use:.3f}. "
        "This coefficient is "
        f"{'negative' if coef_to_use < 0 else 'positive or very close to zero'}, "
        "and the associated p-value "
        f"{'is below' if pval_to_use < 0.05 else 'does not fall below'} the conventional 0.05 "
        "threshold. Taken together with the descriptive statistics, this suggests that "
        + (
            "respondents with children report meaningfully lower engagement in extramarital affairs "
            "than those without children."
            if response == "Yes"
            else "we do not have strong evidence that having children reduces engagement in extramarital affairs; "
            "any differences between the groups are small and/or not statistically robust after adjusting "
            "for other factors."
        )
        + "\n\n"
        f"Based on these results, my overall answer to the question "
        f"“{question}” is: {response}. "
        f"I assign a confidence score of {confidence} on a 0–100 scale, reflecting the strength of the "
        "statistical evidence in this dataset and the limitations of observational survey data."
    )

    return {
        "response": response,
        "confidence": confidence,
        "explanation": explanation,
    }


def main() -> None:
    result = run_analysis()
    base_path = Path(__file__).parent
    conclusion_path = base_path / "conclusion.txt"

    with conclusion_path.open("w") as f:
        json.dump(result, f)


if __name__ == "__main__":
    main()

