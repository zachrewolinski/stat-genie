import json

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def main() -> None:
    df = pd.read_csv("affairs.csv")

    # Basic structure
    print("Columns:", df.columns.tolist())
    print("Head:")
    print(df.head())
    print("\nDescription of key variables:")
    print(df[["affairs", "children", "age", "yearsmarried"]].describe(include="all"))

    # Create binary indicator for any affair
    df["affair_any"] = (df["affairs"] > 0).astype(int)
    print("\nAffair_any value counts:")
    print(df["affair_any"].value_counts(normalize=True))

    # Group statistics by children
    print("\nGroup statistics by children:")
    group_stats = (
        df.groupby("children")
        .agg(
            mean_affairs=("affairs", "mean"),
            mean_affair_any=("affair_any", "mean"),
            count=("affairs", "size"),
        )
    )
    print(group_stats)

    # Encode children as binary for modeling: 1 = yes, 0 = no
    df["children_binary"] = df["children"].map({"no": 0, "yes": 1})

    # Unadjusted logistic regression: any affair ~ children
    print("\nUnadjusted logistic regression: affair_any ~ children_binary")
    model_unadj = smf.logit("affair_any ~ children_binary", data=df).fit(disp=False)
    print(model_unadj.summary())

    # Adjusted logistic regression with key covariates
    print(
        "\nAdjusted logistic regression: affair_any ~ children_binary + age + yearsmarried + religiousness + education + occupation + rating"
    )
    model_adj = smf.logit(
        "affair_any ~ children_binary + age + yearsmarried + religiousness + education + occupation + rating",
        data=df,
    ).fit(disp=False)
    print(model_adj.summary())

    # Extract key statistics for explanation
    prop_any_affair = df["affair_any"].mean()
    stats_no_children = group_stats.loc["no"]
    stats_yes_children = group_stats.loc["yes"]

    coef_unadj = float(model_unadj.params["children_binary"])
    p_unadj = float(model_unadj.pvalues["children_binary"])
    or_unadj = float(np.exp(coef_unadj))

    coef_adj = float(model_adj.params["children_binary"])
    p_adj = float(model_adj.pvalues["children_binary"])
    or_adj = float(np.exp(coef_adj))

    # Based on the evidence, answer the question:
    # "Does having children decrease engagement in extramarital affairs?"
    # The data show higher raw affair rates for parents and an odds ratio > 1
    # in both unadjusted and adjusted models, with only the unadjusted effect
    # reaching conventional statistical significance.
    response = 10  # strong "No" to the hypothesis that children decrease affairs

    explanation = (
        "Research question: Does having children decrease engagement in extramarital affairs?\n\n"
        f"Dataset: 601 first-marriage individuals from the classic Fair affairs data. Overall, "
        f"about {prop_any_affair*100:.1f}% reported at least one extramarital affair in the last year.\n\n"
        f"Descriptively, people without children had an average of {stats_no_children.mean_affairs:.2f} affairs "
        f"and {stats_no_children.mean_affair_any*100:.1f}% had any affair, while people with children had an "
        f"average of {stats_yes_children.mean_affairs:.2f} affairs and {stats_yes_children.mean_affair_any*100:.1f}% "
        "had any affair. Thus, in the raw data, parents engage in extramarital affairs more often, not less.\n\n"
        f"In an unadjusted logistic regression of 'any affair' on a binary children indicator, having children "
        f"is associated with a log-odds coefficient of {coef_unadj:.3f}, corresponding to an odds ratio of "
        f"{or_unadj:.2f} (p = {p_unadj:.3f}). This means that, without adjusting for other factors, individuals "
        "with children have a little more than double the odds of having an affair compared to those without children.\n\n"
        f"After adjusting for age, years married, religiousness, education, occupation, and marriage rating, the "
        f"children coefficient remains positive at {coef_adj:.3f}, with an odds ratio of {or_adj:.2f}, but its "
        f"p-value rises to {p_adj:.3f}, which is not statistically significant at the 0.05 level. The adjusted model "
        "therefore does not provide strong evidence that having children meaningfully changes the odds of affairs, "
        "but it also does not support a protective (decreasing) effect.\n\n"
        "Taken together, the descriptive statistics and regression models provide no evidence that having children "
        "decreases engagement in extramarital affairs; if anything, the association is in the opposite direction "
        "and only modestly attenuated after adjustment. Therefore I answer 'No' to the research question, with a "
        "Likert-scale score of 10 out of 100, reflecting strong confidence that the data do not support a "
        "decrease in affairs among people with children."
    )

    conclusion = {"response": response, "explanation": explanation}
    with open("conclusion.txt", "w", encoding="utf-8") as f:
        json.dump(conclusion, f)


if __name__ == "__main__":
    main()

