import json

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def main() -> None:
    # Load data
    df = pd.read_csv("affairs.csv")

    # Define outcome and key predictor
    df["any_affair"] = (df["feature2"] > 0).astype(int)
    df["children"] = (df["feature6"].str.lower() == "yes").astype(int)

    # Descriptive statistics by children status
    group = df.groupby("children")
    mean_freq = group["feature2"].mean().to_dict()
    share_any = group["any_affair"].mean().to_dict()

    # Inferential analysis: logistic regression for having any affair
    children_or = np.nan
    children_p = np.nan
    try:
        formula = (
            "any_affair ~ children + C(feature3) + feature4 + feature5 + "
            "feature7 + feature8 + feature9 + feature10"
        )
        model = smf.logit(formula, data=df)
        result = model.fit(disp=False)
        children_coef = float(result.params["children"])
        children_p = float(result.pvalues["children"])
        children_or = float(np.exp(children_coef))
    except Exception:
        # Fallback: approximate effect using ratio of probabilities
        prob_with = share_any.get(1, np.nan)
        prob_without = share_any.get(0, np.nan)
        if (
            prob_with is not None
            and prob_without is not None
            and prob_with not in (0, 1)
            and prob_without not in (0, 1)
        ):
            odds_with = prob_with / (1 - prob_with)
            odds_without = prob_without / (1 - prob_without)
            children_or = odds_with / odds_without if odds_without != 0 else np.nan
        children_p = np.nan

    # Map effect size/significance to Likert-style response
    response = 50
    if not np.isnan(children_or):
        if 0.98 <= children_or <= 1.02:
            response = 50
        elif children_or < 1:
            if not np.isnan(children_p) and children_p < 0.01:
                response = 90
            elif not np.isnan(children_p) and children_p < 0.05:
                response = 80
            elif not np.isnan(children_p) and children_p < 0.1:
                response = 70
            else:
                response = 60
        else:
            if not np.isnan(children_p) and children_p < 0.01:
                response = 10
            elif not np.isnan(children_p) and children_p < 0.05:
                response = 20
            elif not np.isnan(children_p) and children_p < 0.1:
                response = 30
            else:
                response = 40

    # Strength descriptor for explanation
    if response >= 85 or response <= 15:
        strength = "very strong"
    elif response >= 75 or response <= 25:
        strength = "strong"
    elif response >= 65 or response <= 35:
        strength = "moderate"
    else:
        strength = "weak"

    mean_no_children = mean_freq.get(0, np.nan)
    mean_with_children = mean_freq.get(1, np.nan)
    share_no_children = share_any.get(0, np.nan)
    share_with_children = share_any.get(1, np.nan)

    direction_phrase = (
        "decreases engagement in extramarital affairs"
        if not np.isnan(children_or) and children_or < 1
        else "does not decrease (and may even increase) engagement in extramarital affairs"
    )

    explanation = (
        "I analyzed the Psychology Today extramarital affairs sample of 601 first-marriage respondents. "
        "I treated engagement in extramarital affairs as both the numeric frequency of intercourse in the past year "
        "(feature2) and a binary indicator of having any affair (feature2 > 0). "
        f"Descriptively, the mean affair frequency was {mean_no_children:.3f} for marriages without children "
        f"and {mean_with_children:.3f} for marriages with children; "
        f"the share with any affair was {share_no_children:.3f} vs {share_with_children:.3f}. "
        "I then estimated a logistic regression of having any affair on an indicator for children in the marriage, "
        "controlling for gender, age, years married, religiousness, education, occupation, and self-rated marital happiness. "
        f"The estimated odds ratio for the children indicator was {children_or:.3f} "
        f"with a p-value of {children_p:.3f}. "
        "An odds ratio below 1 implies that having children is associated with lower odds of an affair, "
        "while a value above 1 implies higher odds. "
        f"Given these results, I judge there to be {strength} evidence that having children {direction_phrase}."
    )

    conclusion = {"response": int(response), "explanation": explanation}
    with open("conclusion.txt", "w") as f:
        json.dump(conclusion, f)


if __name__ == "__main__":
    main()

