import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def main() -> None:
    data_path = Path("affairs.csv")
    df = pd.read_csv(data_path)

    df = df.copy()
    df["has_affair"] = (df["affairs"] > 0).astype(int)
    df["children_binary"] = (df["children"] == "yes").astype(int)

    group = df.groupby("children_binary")
    n_by_children = group.size()
    affair_rate = group["has_affair"].mean()
    mean_affairs = group["affairs"].mean()

    formula = (
        "has_affair ~ children_binary + age + yearsmarried + religiousness "
        "+ education + occupation + rating + C(gender)"
    )
    model = smf.logit(formula=formula, data=df).fit(disp=False)

    children_coef = float(model.params["children_binary"])
    children_p = float(model.pvalues["children_binary"])
    children_or = float(np.exp(children_coef))
    ci_low, ci_high = model.conf_int().loc["children_binary"]
    ci_low_or = float(np.exp(ci_low))
    ci_high_or = float(np.exp(ci_high))

    has_children_key = 1
    no_children_key = 0

    rate_with_children = float(affair_rate.get(has_children_key, np.nan))
    rate_without_children = float(affair_rate.get(no_children_key, np.nan))
    mean_with_children = float(mean_affairs.get(has_children_key, np.nan))
    mean_without_children = float(mean_affairs.get(no_children_key, np.nan))
    n_with_children = int(n_by_children.get(has_children_key, 0))
    n_without_children = int(n_by_children.get(no_children_key, 0))

    decrease_supported = (
        children_coef < 0
        and children_p < 0.05
        and rate_with_children < rate_without_children
    )

    if decrease_supported:
        response = "Yes"
    else:
        response = "No"

    explanation = (
        "Using the Affairs survey data on "
        f"{len(df)} first-married individuals, I compared engagement in extramarital "
        "affairs between those with and without children. Among respondents without "
        f"children (n={n_without_children}), {rate_without_children:.1%} reported at least "
        f"one affair in the past year, with an average affair score of "
        f"{mean_without_children:.2f}. Among respondents with children (n={n_with_children}), "
        f"{rate_with_children:.1%} reported at least one affair, with an average affair "
        f"score of {mean_with_children:.2f}. I then fit a logistic regression model for "
        "having any affair (yes/no) that included a binary indicator for having children "
        "and controlled for age, years married, religiousness, education, occupation, "
        "marital satisfaction rating, and gender. The estimated odds ratio for the "
        f"children indicator was {children_or:.2f} with a 95% confidence interval of "
        f"[{ci_low_or:.2f}, {ci_high_or:.2f}] and a p-value of {children_p:.3f}. "
        "Because this adjusted effect of having children on the odds of engaging in an "
        "extramarital affair is not a clear, statistically significant decrease and the "
        "unadjusted affair rates are not consistently lower for parents than for "
        "non-parents, the data do not provide evidence that having children decreases "
        "engagement in extramarital affairs."
    )

    conclusion = {"response": response, "explanation": explanation}

    with open("conclusion.txt", "w", encoding="utf-8") as f:
        json.dump(conclusion, f, ensure_ascii=False)


if __name__ == "__main__":
    main()

