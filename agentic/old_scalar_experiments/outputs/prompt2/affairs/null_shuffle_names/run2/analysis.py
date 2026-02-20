import json

import numpy as np
import pandas as pd
import statsmodels.api as sm


def main() -> None:
    # Load data
    df = pd.read_csv("affairs.csv")

    # Map columns to their semantic meanings using info.json metadata
    # age -> frequency of extramarital intercourse in past year
    df["affair_freq"] = df["age"].astype(float)

    # religiousness -> indicator of whether there are children in the marriage (yes/no)
    df["has_children"] = df["religiousness"].map({"yes": 1, "no": 0})

    # Additional covariates based on metadata
    # occupation -> age in years (coded numeric)
    df["age_years"] = df["occupation"].astype(float)

    # children -> years married (coded numeric)
    df["yrs_married"] = df["children"].astype(float)

    # rating -> religiousness score
    df["religiousness_score"] = df["rating"].astype(float)

    # yearsmarried -> education level
    df["education_level"] = df["yearsmarried"].astype(float)

    # rownames -> occupation code
    df["occupation_code"] = df["rownames"].astype(float)

    # affairs -> self-rated marriage quality
    df["marriage_rating"] = df["affairs"].astype(float)

    # Binary indicator of any extramarital affair
    df["any_affair"] = (df["affair_freq"] > 0).astype(int)

    # Drop any rows with missing values in key fields (should be rare or none)
    model_cols = [
        "any_affair",
        "has_children",
        "affair_freq",
        "age_years",
        "yrs_married",
        "religiousness_score",
        "education_level",
        "occupation_code",
        "marriage_rating",
    ]
    df_model = df[model_cols].dropna().copy()

    # Descriptive comparison: parents vs non-parents
    group_affair_freq = df_model.groupby("has_children")["affair_freq"]
    group_any_affair = df_model.groupby("has_children")["any_affair"]

    # It is safe to assume both groups exist in this dataset, but guard just in case
    has_children_values = sorted(df_model["has_children"].unique())
    if has_children_values != [0, 1]:
        # If data is degenerate (all with or all without children), we cannot answer confidently
        response = "No"
        confidence = 40
        explanation = (
            "All respondents in the dataset fall into a single group with respect "
            "to having children, so it is not possible to empirically assess how "
            "having children affects engagement in extramarital affairs."
        )
        write_conclusion(response, confidence, explanation)
        return

    mean_affair_no_children = float(group_affair_freq.mean().get(0, np.nan))
    mean_affair_with_children = float(group_affair_freq.mean().get(1, np.nan))

    prop_any_no_children = float(group_any_affair.mean().get(0, np.nan))
    prop_any_with_children = float(group_any_affair.mean().get(1, np.nan))

    # Logistic regression: any affair ~ has_children + controls
    X = df_model[
        [
            "has_children",
            "age_years",
            "yrs_married",
            "religiousness_score",
            "education_level",
            "occupation_code",
            "marriage_rating",
        ]
    ]
    X = sm.add_constant(X, has_constant="add")
    y = df_model["any_affair"]

    logit_model = sm.Logit(y, X)
    result = logit_model.fit(disp=False)

    coef_children = float(result.params["has_children"])
    pvalue_children = float(result.pvalues["has_children"])

    # Decide on answer and confidence
    children_reduce_affairs = (
        (coef_children < 0)
        and (prop_any_with_children <= prop_any_no_children)
        and (mean_affair_with_children <= mean_affair_no_children)
    )

    if children_reduce_affairs and pvalue_children < 0.05:
        response = "Yes"
        confidence = 85
    elif children_reduce_affairs:
        response = "Yes"
        confidence = 65
    else:
        response = "No"
        if (coef_children > 0 and pvalue_children < 0.05) or (
            prop_any_with_children >= prop_any_no_children
            and mean_affair_with_children >= mean_affair_no_children
        ):
            confidence = 80
        else:
            confidence = 55

    explanation = (
        "Using the Psychology Today marital survey data (601 respondents), I treated the 'age' "
        "column as the frequency of extramarital sexual intercourse in the past year and the "
        "'religiousness' column (yes/no) as indicating whether there are children in the marriage, "
        "following the dataset metadata. I created a binary outcome for having any extramarital "
        "affair and compared couples with and without children. The mean affair frequency was "
        f"{mean_affair_no_children:.3f} for couples without children versus "
        f"{mean_affair_with_children:.3f} for couples with children, and the proportion reporting "
        f"any affair was {prop_any_no_children:.3f} without children versus "
        f"{prop_any_with_children:.3f} with children. I then fit a logistic regression model of "
        "any affair on having children, controlling for age, years married, religiousness, "
        "education, occupation, and self-rated marriage quality. The coefficient for having "
        f"children was {coef_children:.3f} with a p-value of {pvalue_children:.3f}, indicating "
        f"{'lower' if coef_children < 0 else 'higher' if coef_children > 0 else 'no clear change in'} "
        "odds of engaging in any extramarital affair among couples with children. Combining the "
        "group comparisons with this regression result leads to the conclusion that "
        f"{'having children is associated with decreased engagement in extramarital affairs.' if response == 'Yes' else 'the data do not show a clear decreasing effect of having children on engagement in extramarital affairs.'}"
    )

    write_conclusion(response, confidence, explanation)


def write_conclusion(response: str, confidence: int, explanation: str) -> None:
    conclusion = {
        "response": response,
        "confidence": int(confidence),
        "explanation": explanation,
    }
    # Write ONLY the JSON object, no extra lines
    with open("conclusion.txt", "w", encoding="utf-8") as f:
        f.write(json.dumps(conclusion))


if __name__ == "__main__":
    main()
