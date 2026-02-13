import json

import numpy as np
import pandas as pd
import statsmodels.api as sm


def main() -> None:
    # Load data
    df = pd.read_csv("affairs.csv")

    # Define outcome and main exposure
    df["affair_any"] = (df["affairs"] > 0).astype(int)
    df["has_children"] = df["children"].str.lower().eq("yes").astype(int)

    # Descriptive statistics by children status
    group = df.groupby("has_children")
    prop_any = group["affair_any"].mean()
    mean_affairs = group["affairs"].mean()
    counts = group["affair_any"].agg(["sum", "count"])

    # Ensure both groups are present
    if set(prop_any.index.tolist()) != {0, 1}:
        raise ValueError("Expected both groups: with and without children.")

    # Logistic regression adjusting for key covariates
    X = df[
        [
            "has_children",
            "age",
            "yearsmarried",
            "religiousness",
            "education",
            "occupation",
            "rating",
        ]
    ].copy()
    X = sm.add_constant(X, has_constant="add")
    y = df["affair_any"]

    model = sm.Logit(y, X, missing="drop")
    result = model.fit(disp=False)

    coef = float(result.params["has_children"])
    se = float(result.bse["has_children"])
    pval = float(result.pvalues["has_children"])
    odds_ratio = float(np.exp(coef))
    ci_low = float(np.exp(coef - 1.96 * se))
    ci_high = float(np.exp(coef + 1.96 * se))

    # Map evidence to 0–100 response scale
    if pval < 0.001:
        response = 90 if odds_ratio < 1 else 10
    elif pval < 0.01:
        response = 80 if odds_ratio < 1 else 20
    elif pval < 0.05:
        response = 70 if odds_ratio < 1 else 30
    else:
        response = 60 if odds_ratio < 1 else 40

    response = int(response)

    # Collect descriptive stats
    prop_children = float(prop_any.loc[1])
    prop_nochildren = float(prop_any.loc[0])
    mean_affairs_children = float(mean_affairs.loc[1])
    mean_affairs_nochildren = float(mean_affairs.loc[0])
    n_children = int(counts.loc[1, "count"])
    n_nochildren = int(counts.loc[0, "count"])
    n_any_children = int(counts.loc[1, "sum"])
    n_any_nochildren = int(counts.loc[0, "sum"])

    # Interpret regression effect
    if odds_ratio < 1:
        if pval < 0.05:
            effect_interpretation = (
                "Because the odds ratio is below 1 and the p-value is below 0.05, "
                "the data provide evidence that people with children have lower odds "
                "of engaging in extramarital affairs after adjustment."
            )
        else:
            effect_interpretation = (
                "Although the odds ratio is below 1, the 95% confidence interval includes 1 "
                "and the p-value is above 0.05, so the evidence for a protective effect of "
                "having children is weak."
            )
    else:
        if pval < 0.05:
            effect_interpretation = (
                "Because the odds ratio is above 1 and the p-value is below 0.05, "
                "the data provide evidence that people with children have higher (not lower) "
                "odds of engaging in extramarital affairs after adjustment."
            )
        else:
            effect_interpretation = (
                "The odds ratio is above 1, but the 95% confidence interval includes 1 "
                "and the p-value is above 0.05, so there is no statistically clear evidence "
                "that having children changes the odds of an affair."
            )

    # Overall conclusion wording
    if odds_ratio < 1 and pval < 0.05:
        overall_conclusion = (
            "Given this pattern, I interpret the evidence as supporting the claim that "
            "having children is associated with a decrease in engagement in extramarital affairs."
        )
    elif odds_ratio < 1 and pval >= 0.05:
        overall_conclusion = (
            "Overall, the data are at most weakly consistent with a decrease in engagement in "
            "extramarital affairs among respondents with children, and the protective effect, if "
            "it exists, is uncertain."
        )
    else:
        overall_conclusion = (
            "Overall, the data do not support the claim that having children decreases engagement "
            "in extramarital affairs; if anything, the association is neutral or in the opposite "
            "direction."
        )

    explanation = (
        "I analyzed the 1969 Psychology Today survey data on 601 first-marriage respondents to test "
        "whether having children decreases engagement in extramarital affairs. "
        "I defined engagement as reporting any extramarital intercourse in the past year and "
        "compared respondents with and without children. "
        f"Among people without children (n={n_nochildren}), {n_any_nochildren} "
        f"({prop_nochildren * 100:.1f}%) reported at least one affair in the past year, with an "
        f"average affairs score of {mean_affairs_nochildren:.2f}. "
        f"Among people with children (n={n_children}), {n_any_children} "
        f"({prop_children * 100:.1f}%) reported at least one affair, with an average affairs score "
        f"of {mean_affairs_children:.2f}. "
        "These descriptive statistics summarize how the prevalence and frequency of affairs differ "
        "between respondents with and without children. "
        "To adjust for potential confounding factors (age, years married, religiousness, education, "
        "occupation, and self-rated marital happiness), I fitted a multivariable logistic regression "
        "model with any affair as the outcome and an indicator for having children as the main "
        "predictor. "
        f"In this model, the coefficient for having children corresponds to an odds ratio of "
        f"{odds_ratio:.2f} (95% CI {ci_low:.2f}–{ci_high:.2f}, p={pval:.3g}). "
        f"{effect_interpretation} "
        f"{overall_conclusion} "
        "On a 0–100 scale where 0 is a strong 'No' and 100 is a strong 'Yes' answer to the question "
        "'Does having children decrease engagement in extramarital affairs?', I assign a score of "
        f"{response}, reflecting the strength and direction of the regression evidence together with "
        "the descriptive comparisons."
    )

    conclusion = {"response": response, "explanation": explanation}

    # Write required JSON output
    with open("conclusion.txt", "w") as f:
        json.dump(conclusion, f)


if __name__ == "__main__":
    main()

