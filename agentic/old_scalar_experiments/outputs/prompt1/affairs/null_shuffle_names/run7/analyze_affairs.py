import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def main() -> None:
    data_path = Path("affairs.csv")
    if not data_path.exists():
        raise FileNotFoundError("affairs.csv not found in current directory.")

    df = pd.read_csv(data_path)

    # According to info.json, the column named "age" encodes frequency of
    # extramarital sexual intercourse in the past year, and "religiousness"
    # is a yes/no indicator for whether there are children in the marriage.
    affairs_freq = df["age"].astype(float)
    has_children = (df["religiousness"].str.lower() == "yes").astype(int)

    # Basic descriptive statistics by children status.
    grouped = affairs_freq.groupby(has_children)
    mean_no_children = grouped.mean().get(0, np.nan)
    mean_with_children = grouped.mean().get(1, np.nan)
    n_no_children = grouped.count().get(0, 0)
    n_with_children = grouped.count().get(1, 0)

    # Poisson regression of affair frequency on children indicator.
    X = sm.add_constant(has_children)
    model = sm.GLM(affairs_freq, X, family=sm.families.Poisson())
    result = model.fit()

    coef_children = float(result.params["religiousness"])
    pvalue_children = float(result.pvalues["religiousness"])
    irr_children = float(np.exp(coef_children))

    # Decision rule: if having children is associated with a statistically
    # significant reduction in expected affair frequency at the 5% level,
    # answer "Yes"; otherwise answer "No".
    decreases_affairs = coef_children < 0 and pvalue_children < 0.05
    response = "Yes" if decreases_affairs else "No"

    percent_change = (irr_children - 1.0) * 100.0

    explanation = (
        "Using the Psychology Today 1969 survey of 601 first-marriage respondents, "
        "I analyzed whether having children is associated with engagement in extramarital affairs. "
        "Based on the metadata in info.json, I treated the 'age' column as the numeric frequency of "
        "extramarital sexual intercourse during the past year and the 'religiousness' column as a "
        "yes/no indicator of whether there are children in the marriage. "
        f"In the data, couples without children (n={n_no_children}) had a mean affair-frequency score "
        f"of {mean_no_children:.3f}, whereas couples with children (n={n_with_children}) had a mean of "
        f"{mean_with_children:.3f}. "
        "To formally test the relationship, I fit a Poisson regression of affair frequency on a binary "
        "indicator for having children. "
        f"The estimated coefficient on the children indicator corresponds to an incidence-rate ratio of "
        f"{irr_children:.3f} (approximately {percent_change:+.1f}% change in expected affair frequency for "
        "couples with children relative to those without), with a p-value of "
        f"{pvalue_children:.3g}. "
        "Based on this model and a 5% significance threshold, "
        "I therefore conclude that having children "
        + ("is associated with a statistically significant decrease in engagement in extramarital affairs."
           if decreases_affairs
           else "does not show a statistically significant decreasing effect on engagement in extramarital affairs.")
        + " This conclusion is correlational and does not establish causality, and it does not adjust for "
        "other potential confounders such as years married, age, education, or marital satisfaction."
    )

    conclusion = {"response": response, "explanation": explanation}

    with open("conclusion.txt", "w", encoding="utf-8") as f:
        json.dump(conclusion, f, ensure_ascii=False)


if __name__ == "__main__":
    main()

