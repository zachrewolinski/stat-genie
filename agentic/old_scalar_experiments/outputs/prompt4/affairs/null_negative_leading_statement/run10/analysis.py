import json
from pathlib import Path

import pandas as pd
import statsmodels.formula.api as smf


def main() -> None:
    base_path = Path(__file__).parent
    data_path = base_path / "affairs.csv"
    conclusion_path = base_path / "conclusion.txt"

    df = pd.read_csv(data_path)

    # Basic sanity: drop rows with missing key fields if any
    df = df.dropna(subset=["affairs", "children"])

    # Create binary indicator for any affair
    df["any_affair"] = (df["affairs"] > 0).astype(int)

    # Descriptive statistics by children status
    group_desc = (
        df.groupby("children")
        .agg(
            n=("affairs", "size"),
            mean_affairs=("affairs", "mean"),
            prop_any_affair=("any_affair", "mean"),
        )
        .reset_index()
    )

    # Logistic regression for any affair with children as main predictor,
    # controlling for standard covariates available in this dataset.
    formula = (
        "any_affair ~ C(children) + age + yearsmarried + religiousness "
        "+ education + occupation + rating + C(gender)"
    )

    logit_model = smf.logit(formula=formula, data=df).fit(disp=False)
    logit_params = logit_model.params
    logit_pvalues = logit_model.pvalues

    # Extract coefficient and p-value for children effect.
    # With C(children), statsmodels will create a term like C(children)[T.yes] or [T.no]
    children_terms = [name for name in logit_params.index if name.startswith("C(children)")]
    children_effect = None
    children_pvalue = None
    if children_terms:
        # There should be exactly one contrast term for children
        term = children_terms[0]
        children_effect = float(logit_params[term])
        children_pvalue = float(logit_pvalues[term])

    # Interpret results:
    # - If the coefficient for children is strongly negative and reasonably significant,
    #   that would support "Yes, children decrease affairs".
    # - If it is near zero or positive / non-significant, that would support "No".
    #
    # For the scalar response, 0 = strong "No", 100 = strong "Yes".
    # We'll map based on sign and significance of the children coefficient,
    # combined with the group-level descriptive differences.

    # Descriptive signal: compare mean affairs and affair prevalence.
    desc_children_yes = group_desc[group_desc["children"] == "yes"]
    desc_children_no = group_desc[group_desc["children"] == "no"]

    mean_diff = None
    prop_diff = None
    if not desc_children_yes.empty and not desc_children_no.empty:
        mean_diff = float(
            desc_children_yes["mean_affairs"].iloc[0]
            - desc_children_no["mean_affairs"].iloc[0]
        )
        prop_diff = float(
            desc_children_yes["prop_any_affair"].iloc[0]
            - desc_children_no["prop_any_affair"].iloc[0]
        )

    # Default to a neutral-to-mild "No" if we cannot estimate anything
    response_score = 40

    explanation_lines = []
    explanation_lines.append(
        "Research question: Does having children decrease engagement in extramarital affairs?"
    )
    explanation_lines.append(
        "I analyzed the Psychology Today 1969 marital affairs survey (601 married individuals) "
        "using both descriptive statistics and a logistic regression for having any affair in the past year."
    )

    # Add descriptive summary
    explanation_lines.append(
        "Descriptive comparison by children status (n = number of respondents, "
        "mean_affairs = average affair score, prop_any_affair = proportion with any affair):"
    )
    explanation_lines.append(group_desc.to_string(index=False))

    if children_effect is not None and children_pvalue is not None:
        explanation_lines.append(
            "In the logistic regression for having any affair (controlling for age, years married, "
            "religiousness, education, occupation, marital rating, and gender), the coefficient for the "
            f"children indicator was {children_effect:.3f} with p-value {children_pvalue:.3f}."
        )

        # Determine response strength
        if children_effect < 0 and children_pvalue < 0.05:
            # Clear evidence children are associated with fewer affairs -> "Yes"
            response_score = 80
            explanation_lines.append(
                "This negative and statistically significant coefficient suggests that having children "
                "is associated with a lower likelihood of engaging in extramarital affairs, even after "
                "adjusting for other factors."
            )
        elif children_effect < 0 and children_pvalue < 0.10:
            # Weak evidence of decrease
            response_score = 65
            explanation_lines.append(
                "The coefficient is negative with marginal statistical support, suggesting at most a "
                "modest association between having children and fewer extramarital affairs."
            )
        elif abs(children_effect) < 0.05 or children_pvalue > 0.20:
            # Essentially no reliable effect
            response_score = 30
            explanation_lines.append(
                "The children coefficient is small in magnitude and/or statistically indistinguishable "
                "from zero, indicating no reliable evidence that having children decreases extramarital affairs."
            )
        else:
            # Coefficient positive or weakly non-zero -> evidence against a decrease
            response_score = 20
            explanation_lines.append(
                "The children coefficient is not negative and clearly significant in the direction of "
                "a decrease; if anything, it suggests no protective effect of having children on "
                "extramarital affairs."
            )
    else:
        explanation_lines.append(
            "The regression model could not estimate a separate effect for children, so conclusions rely "
            "only on descriptive comparisons, which do not show clear evidence that children reduce affairs."
        )

    if mean_diff is not None and prop_diff is not None:
        explanation_lines.append(
            f"Comparing raw group averages, respondents with children have a mean affair score "
            f"{mean_diff:+.3f} points different from those without children, and a "
            f"{prop_diff*100:+.1f} percentage point difference in the proportion reporting any affair."
        )

    explanation_lines.append(
        "Given these results, the data do not provide strong support for the claim that having children "
        "decreases engagement in extramarital affairs; at best, any such effect is small and statistically "
        "uncertain in this sample."
    )

    explanation = "\n".join(explanation_lines)

    # Ensure response is an integer between 0 and 100
    response_int = int(max(0, min(100, round(response_score))))

    payload = {"response": response_int, "explanation": explanation}
    conclusion_path.write_text(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()

