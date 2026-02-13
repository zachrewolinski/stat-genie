import json
from pathlib import Path

import pandas as pd
import statsmodels.formula.api as smf


DATA_FILE = Path("affairs.csv")
OUTPUT_FILE = Path("conclusion.txt")


def main() -> None:
    df = pd.read_csv(DATA_FILE)

    # Based on info.json metadata, columns are semantically remapped:
    # - age: coded frequency of extramarital affairs (0, 1, 2, 3, 7, 12)
    # - religiousness: yes/no indicator for whether there are children
    # Other columns provide demographic and relationship context.

    df = df.copy()

    # Core variables for the research question
    df["affair_score"] = df["age"]
    df["has_children"] = df["religiousness"].map({"yes": 1, "no": 0})
    df = df.dropna(subset=["affair_score", "has_children"])
    df["any_affair"] = (df["affair_score"] > 0).astype(int)

    # Additional covariates (semantic remapping from metadata)
    df["age_years"] = df["occupation"]  # categorical age bands coded numerically
    df["years_married"] = df["children"]  # coded years married
    df["religiousness_level"] = df["rating"]  # 1 (anti) – 5 (very religious)
    df["education_years"] = df["yearsmarried"]  # 9–20 years of education
    df["occupation_code"] = df["rownames"]  # 1–7 occupation code
    df["marriage_rating"] = df["affairs"]  # 1 (very unhappy) – 5 (very happy)

    # Unadjusted comparisons: affair intensity and any-affair prevalence
    group_stats = df.groupby("has_children")["affair_score"].agg(
        ["mean", "median", "std", "count"]
    )
    any_affair_rates = df.groupby("has_children")["any_affair"].mean()

    with_children_mean = float(group_stats.loc[1, "mean"])
    without_children_mean = float(group_stats.loc[0, "mean"])
    with_children_rate = float(any_affair_rates.loc[1])
    without_children_rate = float(any_affair_rates.loc[0])

    # Adjusted analysis: logistic regression for having any affair
    coef_children = None
    p_children = None

    try:
        model = smf.logit(
            "any_affair ~ has_children + age_years + years_married + "
            "religiousness_level + education_years + occupation_code + "
            "marriage_rating + C(gender)",
            data=df,
        ).fit(disp=False)
        coef_children = float(model.params["has_children"])
        p_children = float(model.pvalues["has_children"])
    except Exception:
        coef_children = float("nan")
        p_children = float("nan")

    # Decision rule:
    # - If people with children have *lower* affair frequency and lower any-affair rate
    #   and the adjusted effect is significantly negative (p < 0.05), we answer "Yes".
    # - Otherwise (no clear decrease, or increased risk), we answer "No".
    decreases_unadjusted = (with_children_mean < without_children_mean) and (
        with_children_rate < without_children_rate
    )
    decreases_adjusted = (
        (coef_children is not None)
        and not pd.isna(coef_children)
        and not pd.isna(p_children)
        and (coef_children < 0.0)
        and (p_children < 0.05)
    )

    if decreases_unadjusted and decreases_adjusted:
        response = "Yes"
    else:
        response = "No"

    direction_word = "decrease" if coef_children is not None and coef_children < 0 else "increase"

    explanation = (
        "Using the provided sample of married individuals (n={n}), we recode the survey so that "
        "the variable 'age' captures the coded frequency of extramarital sexual intercourse over "
        "the past year and the 'religiousness' yes/no factor indicates whether there are "
        "children in the marriage, as described in the metadata. "
        "We define an affair-intensity score from this variable and a binary indicator for having "
        "any affair (score > 0).\n\n"
        "Unadjusted comparisons show that the mean affair-intensity score is {mean_no:.3f} for "
        "individuals without children and {mean_yes:.3f} for those with children, while the "
        "proportion having at least one affair is {rate_no:.3f} without children versus "
        "{rate_yes:.3f} with children. These summaries indicate that parents "
        "{unadj_direction} extramarital involvement relative to non-parents.\n\n"
        "To account for other factors captured in the dataset, we fit a logistic regression model "
        "for the probability of having any affair, including an indicator for having children "
        "alongside age band, years married, religiousness level, education, occupation code, "
        "marital happiness rating, and gender. In this model, the coefficient for the "
        "children indicator is {coef:.3f} with p-value {pval:.3f}, implying that, after "
        "adjusting for these covariates, having children tends to {direction} the likelihood of "
        "an extramarital affair.\n\n"
        "Given both the unadjusted group comparisons and the adjusted regression result, the "
        "evidence in this dataset supports the conclusion: {response}."
    ).format(
        n=int(len(df)),
        mean_no=without_children_mean,
        mean_yes=with_children_mean,
        rate_no=without_children_rate,
        rate_yes=with_children_rate,
        unadj_direction=(
            "show lower" if with_children_mean < without_children_mean else "do not show lower"
        ),
        coef=coef_children if coef_children is not None else float("nan"),
        pval=p_children if p_children is not None else float("nan"),
        direction=direction_word,
        response=response,
    )

    result = {"response": response, "explanation": explanation}

    with OUTPUT_FILE.open("w") as f:
        json.dump(result, f)


if __name__ == "__main__":
    main()

