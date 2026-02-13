import json

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def main() -> None:
    df = pd.read_csv("affairs.csv")
    df = df.copy()

    df["affair_freq"] = df["feature2"]
    df["affair"] = (df["affair_freq"] > 0).astype(int)
    df["children_yes"] = df["feature6"].str.lower().eq("yes").astype(int)

    group_mean_freq = df.groupby("children_yes")["affair_freq"].mean()
    group_affair_rate = df.groupby("children_yes")["affair"].mean()

    formula = (
        "affair ~ children_yes + feature4 + feature5 + "
        "feature7 + feature8 + feature9 + feature10 + C(feature3)"
    )
    model = smf.logit(formula=formula, data=df).fit(disp=False)

    coef_child = float(model.params["children_yes"])
    pvalue_child = float(model.pvalues["children_yes"])

    mean_covs = df[["feature4", "feature5", "feature7", "feature8", "feature9", "feature10"]].mean()
    mode_gender = df["feature3"].mode().iloc[0]

    base = {
        "feature4": mean_covs["feature4"],
        "feature5": mean_covs["feature5"],
        "feature7": mean_covs["feature7"],
        "feature8": mean_covs["feature8"],
        "feature9": mean_covs["feature9"],
        "feature10": mean_covs["feature10"],
        "feature3": mode_gender,
    }
    row_no_child = base.copy()
    row_no_child["children_yes"] = 0
    row_child = base.copy()
    row_child["children_yes"] = 1

    pred_no_child = float(model.predict(pd.DataFrame([row_no_child]))[0])
    pred_child = float(model.predict(pd.DataFrame([row_child]))[0])
    prob_diff = pred_child - pred_no_child

    if pvalue_child < 0.05:
        if coef_child < 0:
            if abs(prob_diff) >= 0.10:
                response = 90
            else:
                response = 75
        else:
            if abs(prob_diff) >= 0.10:
                response = 10
            else:
                response = 25
    elif pvalue_child < 0.10:
        if coef_child < 0:
            response = 65
        else:
            response = 35
    else:
        if coef_child < 0:
            response = 55
        elif coef_child > 0:
            response = 45
        else:
            response = 50

    response_int = int(max(0, min(100, round(response))))

    mean_freq_no_child = float(group_mean_freq.get(0, np.nan))
    mean_freq_child = float(group_mean_freq.get(1, np.nan))
    rate_no_child = float(group_affair_rate.get(0, np.nan))
    rate_child = float(group_affair_rate.get(1, np.nan))

    explanation = (
        "Research question: Does having children decrease engagement in extramarital affairs? "
        "Using the Fair (1978) affairs dataset with 601 married respondents, I compared people with and without children "
        "on both the frequency of affairs in the past year and the probability of having any affair. "
        f"Descriptively, the mean affair-frequency score (0 = no affairs, higher = more frequent affairs) was "
        f"{mean_freq_child:.2f} for respondents with children and {mean_freq_no_child:.2f} for those without children. "
        f"The share of respondents reporting any extramarital sex in the past year was {rate_child:.2%} with children "
        f"versus {rate_no_child:.2%} without children. "
        "I then fit a logistic regression for having any affair as the outcome, with an indicator for having children "
        "and controls for age, years married, religiousness, education, occupation, gender, and self-rated marital happiness. "
        f"In this model, the coefficient on the 'has children' indicator was {coef_child:.3f} with p-value {pvalue_child:.3f}. "
    )

    if coef_child < 0:
        explanation += (
            "This negative coefficient implies that, holding other factors constant, having children is associated with "
            "a lower probability of having an affair. "
            f"The model's predicted probability of any affair at average covariate values was {pred_no_child:.1%} without "
            f"children and {pred_child:.1%} with children, a difference of {prob_diff:.1%}."
        )
    else:
        explanation += (
            "This positive coefficient implies that, holding other factors constant, having children is associated with "
            "a higher probability of having an affair. "
            f"The model's predicted probability of any affair at average covariate values was {pred_no_child:.1%} without "
            f"children and {pred_child:.1%} with children, a difference of {prob_diff:.1%}."
        )

    explanation += (
        " On a 0-100 scale where 0 means a strong 'No' and 100 a strong 'Yes' to the claim that having children decreases "
        f"engagement in extramarital affairs, I assign a score of {response_int}. "
        "This score reflects both the direction and statistical strength of the estimated effect."
    )

    conclusion = {"response": response_int, "explanation": explanation}

    with open("conclusion.txt", "w", encoding="utf-8") as f:
        json.dump(conclusion, f, ensure_ascii=False)


if __name__ == "__main__":
    main()

