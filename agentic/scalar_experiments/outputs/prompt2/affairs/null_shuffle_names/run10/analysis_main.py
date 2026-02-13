import json
from typing import Dict

import numpy as np
import pandas as pd
import statsmodels.api as sm


def run_analysis() -> Dict[str, object]:
    # Load data
    df = pd.read_csv("affairs.csv")

    # According to info.json metadata:
    # - Column 'age' encodes frequency of extramarital intercourse (0 = none, higher = more)
    # - Column 'religiousness' actually indicates whether there are children in the marriage (\"yes\"/\"no\")
    df["has_children"] = (df["religiousness"] == "yes").astype(int)
    df["any_affair"] = (df["age"] > 0).astype(int)

    # Descriptive comparisons
    mean_affair_freq = df.groupby("has_children")["age"].mean()
    any_affair_rate = df.groupby("has_children")["any_affair"].mean()

    mean_no_children = float(mean_affair_freq.loc[0])
    mean_with_children = float(mean_affair_freq.loc[1])
    rate_no_children = float(any_affair_rate.loc[0])
    rate_with_children = float(any_affair_rate.loc[1])

    # Logistic regression: probability of any affair vs having children
    X = sm.add_constant(df["has_children"])
    y = df["any_affair"]
    logit_model = sm.Logit(y, X)
    result = logit_model.fit(disp=False)

    coef_children = float(result.params["has_children"])
    pval_children = float(result.pvalues["has_children"])
    odds_ratio = float(np.exp(coef_children))

    # Decide on response
    children_reduce_affairs = (
        mean_with_children < mean_no_children and coef_children < 0
    )

    if children_reduce_affairs and pval_children < 0.05:
        response = "Yes"
        # Stronger confidence for smaller p-values
        if pval_children < 0.01:
            confidence = 90
        else:
            confidence = 80
    else:
        response = "No"
        # If effect is clearly not negative (mean_with_children >= mean_no_children),
        # or statistically weak, we conclude there is no clear evidence of a decrease.
        if pval_children < 0.05 and coef_children > 0:
            confidence = 85
        elif pval_children < 0.1:
            confidence = 70
        else:
            confidence = 65

    explanation = (
        "I treated the 'age' column as the coded frequency of extramarital sexual "
        "intercourse (0 = none, higher values = more frequent) and the 'religiousness' "
        "column as an indicator of whether there are children in the marriage, as "
        "described in the metadata. I created a binary variable for having any affair "
        "(age > 0) and compared people with and without children. The mean coded affair "
        f"frequency was {mean_no_children:.2f} for couples without children and "
        f"{mean_with_children:.2f} for couples with children, while the share with any "
        f"affair was {rate_no_children:.3f} without children versus "
        f"{rate_with_children:.3f} with children. I then fit a logistic regression model "
        "of having any affair on a binary indicator for having children. The estimated "
        f"odds ratio for having children was {odds_ratio:.2f} (p-value {pval_children:.3f}). "
        "Because the regression coefficient for having children is "
        f"{'negative' if coef_children < 0 else 'positive'} with this p-value and the "
        "group differences do not provide strong evidence that couples with children have "
        "fewer extramarital affairs, I conclude that the data do not support the claim "
        "that having children decreases engagement in extramarital affairs."
    )

    return {
        "response": response,
        "confidence": confidence,
        "explanation": explanation,
    }


def main() -> None:
    result = run_analysis()
    with open("conclusion.txt", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False)


if __name__ == "__main__":
    main()

