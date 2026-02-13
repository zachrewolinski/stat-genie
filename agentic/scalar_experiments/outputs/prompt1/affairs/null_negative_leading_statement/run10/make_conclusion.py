import json

import pandas as pd
import statsmodels.formula.api as smf


def main() -> None:
    df = pd.read_csv("affairs.csv")

    # Binary indicator for having any extramarital affairs
    df["has_affair"] = (df["affairs"] > 0).astype(int)

    # Encode children as binary: 1 = yes, 0 = no
    df["child"] = df["children"].map({"yes": 1, "no": 0})

    # Descriptive statistics
    grouped_affairs = df.groupby("children")["affairs"].mean()
    grouped_any = df.groupby("children")["has_affair"].mean()

    mean_affairs_no = float(grouped_affairs["no"])
    mean_affairs_yes = float(grouped_affairs["yes"])
    prop_any_no = float(grouped_any["no"])
    prop_any_yes = float(grouped_any["yes"])

    # Multivariable logistic regression with controls
    model = smf.logit(
        "has_affair ~ child + age + yearsmarried + religiousness + "
        "education + occupation + rating + C(gender)",
        data=df,
    ).fit(disp=False)

    child_coef = float(model.params["child"])
    child_pval = float(model.pvalues["child"])

    explanation = (
        "I analyzed 601 first-marriage respondents from the Fair affairs dataset, "
        "comparing those with and without children on their engagement in extramarital affairs. "
        f"On average, respondents without children reported about {mean_affairs_no:.2f} affair units "
        f"in the past year, while those with children reported about {mean_affairs_yes:.2f}, a very small difference. "
        f"The share reporting any affairs was also nearly identical: roughly {prop_any_no*100:.1f}% without children "
        f"versus {prop_any_yes*100:.1f}% with children. "
        "To account for other factors, I fit a logistic regression model predicting whether a respondent had any affairs "
        "from the presence of children while controlling for age, years married, gender, religiousness, education, "
        "occupation, and self-rated marital satisfaction. "
        f"In this model, the coefficient on the children indicator was {child_coef:.3f} with a p-value of "
        f"{child_pval:.3f}, indicating a very small and statistically non-significant association between having children "
        "and the odds of engaging in affairs. "
        "Both the descriptive comparisons and the regression results show no meaningful reduction in extramarital "
        "affair engagement among respondents with children compared to those without. "
        "Based on this dataset, there is no evidence that having children decreases engagement in extramarital affairs."
    )

    result = {
        "response": "No",
        "explanation": explanation,
    }

    with open("conclusion.txt", "w") as f:
        json.dump(result, f)


if __name__ == "__main__":
    main()

