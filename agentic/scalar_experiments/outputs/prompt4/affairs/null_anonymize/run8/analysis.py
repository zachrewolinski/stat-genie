import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def main() -> None:
    # Load metadata and data
    info_path = Path("info.json")
    data_path = Path("affairs.csv")

    with info_path.open() as f:
        info = json.load(f)

    df = pd.read_csv(data_path)

    # Define variables
    # Binary outcome: any extramarital intercourse in past year
    df["affair_any"] = (df["feature2"] > 0).astype(int)
    # Predictor: presence of children in the marriage
    df["has_children"] = (df["feature6"] == "yes").astype(int)

    n = len(df)
    # Descriptive rates
    rate_children = df.loc[df["has_children"] == 1, "affair_any"].mean()
    rate_no_children = df.loc[df["has_children"] == 0, "affair_any"].mean()

    # Logistic regression of affair on children (bivariate to keep focus on question)
    X = sm.add_constant(df["has_children"])
    y = df["affair_any"]
    model = sm.Logit(y, X)
    res = model.fit(disp=False)

    coef_children = res.params["has_children"]
    p_children = res.pvalues["has_children"]
    odds_ratio = float(np.exp(coef_children))

    # Map evidence to Likert-style 0–100 response
    # Interpreting the question as:
    # "Yes" = having children decreases engagement in extramarital affairs.
    if coef_children < 0:
        # Children associated with *lower* odds of affairs
        if p_children < 0.01:
            response = 90
        elif p_children < 0.05:
            response = 80
        elif p_children < 0.1:
            response = 65
        else:
            response = 55
    elif coef_children > 0:
        # Children associated with *higher* odds of affairs
        if p_children < 0.01:
            response = 10
        elif p_children < 0.05:
            response = 20
        elif p_children < 0.1:
            response = 35
        else:
            response = 45
    else:
        # Essentially no estimated effect
        response = 50

    # Build explanation text
    question = info.get("research_questions", [""])[0]

    explanation = (
        f"Research question: {question.strip()} "
        f"Using the Fair affairs survey data (n={n}), I created a binary outcome "
        f"for whether a respondent reported any extramarital intercourse in the past year "
        f"and a binary predictor for whether there are children in the marriage. "
        f"The raw proportion of respondents with any affair was "
        f"{rate_children:.3f} among couples with children and {rate_no_children:.3f} "
        f"among couples without children. "
        f"I then fit a bivariate logistic regression of affair occurrence on the "
        f'\"has children\" indicator. The estimated coefficient for having children was '
        f"{coef_children:.3f}, corresponding to an odds ratio of {odds_ratio:.3f} "
        f"(p-value={p_children:.3f}). "
        f"These results indicate that "
        f"{'having children is associated with lower odds of affairs' if coef_children < 0 else 'having children is not associated with lower odds of affairs'} "
        f"at conventional significance levels. "
        f"On a 0–100 scale where higher values represent a stronger 'Yes' to the statement "
        f"that having children decreases engagement in extramarital affairs, "
        f"I map the statistical evidence to a response of {response}."
    )

    output = {"response": int(response), "explanation": explanation}

    with Path("conclusion.txt").open("w") as f:
        json.dump(output, f)


if __name__ == "__main__":
    main()

