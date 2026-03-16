import json

import numpy as np
import pandas as pd
import statsmodels.api as sm


def main() -> None:
    # Load dataset
    df = pd.read_csv("affairs.csv")

    # Reconstruct semantics from metadata:
    # - "age" encodes frequency of extramarital intercourse in the past year.
    # - "religiousness" is a yes/no indicator for whether there are children in the marriage.
    df["any_affair"] = (df["age"] > 0).astype(int)
    df["has_children"] = df["religiousness"].str.lower().eq("yes").astype(int)

    # Basic proportions
    grouped = (
        df.groupby("has_children")["any_affair"]
        .agg(["mean", "count", "sum"])
        .rename(
            columns={
                "mean": "prop_any_affair",
                "sum": "num_with_affair",
            }
        )
    )

    prop_no_children = float(grouped.loc[0, "prop_any_affair"])
    prop_children = float(grouped.loc[1, "prop_any_affair"])
    n_no_children = int(grouped.loc[0, "count"])
    n_children = int(grouped.loc[1, "count"])
    n_affairs_no_children = int(grouped.loc[0, "num_with_affair"])
    n_affairs_children = int(grouped.loc[1, "num_with_affair"])

    # Logistic regression: any_affair ~ has_children
    X = sm.add_constant(df["has_children"])
    model = sm.Logit(df["any_affair"], X).fit(disp=False)

    coef_children = float(model.params["has_children"])
    se_children = float(model.bse["has_children"])
    p_children = float(model.pvalues["has_children"])
    odds_ratio_children = float(np.exp(coef_children))

    # Summarize evidence: here the p-value is large (~0.78) and the
    # point estimate is very close to no effect (odds ratio near 1),
    # so we conclude there is no statistically supported decrease.
    response = 10

    explanation = (
        "I used the 601 married respondents in this dataset, where the 'age' column "
        "actually encodes how often a respondent engaged in extramarital intercourse "
        "over the past year (values 0, 1, 2, 3, 7, or 12), and the 'religiousness' "
        "column is a yes/no indicator for whether there are children in the marriage. "
        "I defined having any extramarital affair as age > 0 and having children as "
        "religiousness == 'yes'. Among couples without children, "
        f"{n_affairs_no_children}/{n_no_children} respondents ({prop_no_children:.1%}) "
        "reported at least one affair, compared with "
        f"{n_affairs_children}/{n_children} ({prop_children:.1%}) among couples with children, "
        "a difference of only about "
        f"{(prop_children - prop_no_children) * 100:.1f} percentage points. "
        "A logistic regression of any-affair status on the children indicator yielded a "
        f"coefficient of {coef_children:.3f} for having children, corresponding to an odds "
        f"ratio of {odds_ratio_children:.2f} with a large p-value of {p_children:.3f}, "
        "indicating no statistically significant association between having children and "
        "the likelihood of engaging in extramarital affairs. Taken together, these results "
        "provide no meaningful evidence that having children decreases engagement in "
        "extramarital affairs in this sample, so I answer 'No' to the research question."
    )

    result = {
        "response": response,
        "explanation": explanation,
    }

    with open("conclusion.txt", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False)


if __name__ == "__main__":
    main()
