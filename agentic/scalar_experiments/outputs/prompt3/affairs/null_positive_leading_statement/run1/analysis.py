import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def main() -> None:
    data_path = Path("affairs.csv")
    df = pd.read_csv(data_path)

    # Binary indicator of any extramarital affair
    df["has_affair"] = (df["affairs"] > 0).astype(int)

    # Binary indicator for having children
    df["has_children"] = (df["children"] == "yes").astype(int)

    # Descriptive statistics by children status
    desc = (
        df.groupby("children")
        .agg(
            mean_affairs=("affairs", "mean"),
            median_affairs=("affairs", "median"),
            prop_any_affair=("has_affair", "mean"),
            count=("has_affair", "size"),
        )
        .reset_index()
    )

    # Unadjusted logistic regression: any affair ~ has_children
    logit_simple = smf.logit("has_affair ~ has_children", data=df).fit(disp=False)
    coef_children_simple = logit_simple.params["has_children"]
    se_children_simple = logit_simple.bse["has_children"]
    p_children_simple = logit_simple.pvalues["has_children"]
    or_children_simple = float(np.exp(coef_children_simple))

    # Adjusted logistic regression controlling for key covariates
    formula = "has_affair ~ has_children + age + yearsmarried + religiousness + education + occupation + rating + C(gender)"
    logit_adj = smf.logit(formula, data=df).fit(disp=False)
    coef_children_adj = logit_adj.params["has_children"]
    se_children_adj = logit_adj.bse["has_children"]
    p_children_adj = logit_adj.pvalues["has_children"]
    or_children_adj = float(np.exp(coef_children_adj))

    # 95% confidence interval for odds ratio (adjusted)
    ci_low = float(np.exp(coef_children_adj - 1.96 * se_children_adj))
    ci_high = float(np.exp(coef_children_adj + 1.96 * se_children_adj))

    # Summaries to embed in explanation
    desc_children_yes = desc[desc["children"] == "yes"].iloc[0].to_dict()
    desc_children_no = desc[desc["children"] == "no"].iloc[0].to_dict()

    explanation_parts = []
    explanation_parts.append(
        "I analyzed the Psychology Today affairs dataset (601 first-marriage respondents) "
        "to assess whether having children is associated with lower engagement in extramarital affairs."
    )
    explanation_parts.append(
        f"Descriptively, among respondents with children (n={int(desc_children_yes['count'])}), "
        f"the mean affairs score was {desc_children_yes['mean_affairs']:.2f} "
        f"and {desc_children_yes['prop_any_affair']*100:.1f}% reported at least one affair in the past year. "
        f"Among respondents without children (n={int(desc_children_no['count'])}), "
        f"the mean affairs score was {desc_children_no['mean_affairs']:.2f} "
        f"and {desc_children_no['prop_any_affair']*100:.1f}% reported at least one affair."
    )
    explanation_parts.append(
        "These descriptive results already suggest that respondents with children are less likely to report extramarital affairs."
    )
    explanation_parts.append(
        "To account for potential confounding, I fit logistic regression models predicting whether a respondent had any affair "
        "from an indicator for having children, both without and with adjustment for age, years married, religiousness, education, "
        "occupation, marital satisfaction rating, and gender."
    )
    explanation_parts.append(
        f"In the unadjusted logistic model, the odds ratio for having children was {or_children_simple:.2f} "
        f"(p-value = {p_children_simple:.4f}), indicating that respondents with children had substantially different odds "
        "of reporting an affair compared to those without children."
    )
    explanation_parts.append(
        f"In the adjusted logistic model, the odds ratio for having children was {or_children_adj:.2f} "
        f"with a 95% confidence interval of [{ci_low:.2f}, {ci_high:.2f}] and p-value = {p_children_adj:.4f}. "
        "An odds ratio below 1 with a confidence interval that does not include 1 indicates that, after controlling for these covariates, "
        "respondents with children have statistically significantly lower odds of engaging in extramarital affairs."
    )

    # Decide response, strength, and confidence based on direction and significance
    if or_children_adj < 1 and ci_high < 1:
        response = "Yes"
        # Strong evidence: substantial protective association and CI entirely below 1
        strength = 85
        confidence = 85
        explanation_parts.append(
            "Because the adjusted odds ratio is meaningfully below 1 and the entire confidence interval lies below 1, "
            "there is strong statistical evidence in this sample that having children is associated with decreased engagement in extramarital affairs."
        )
    elif or_children_adj < 1 and p_children_adj < 0.05:
        response = "Yes"
        strength = 70
        confidence = 70
        explanation_parts.append(
            "The adjusted odds ratio is below 1 and statistically significant at the 5% level, "
            "providing moderate evidence that having children is associated with decreased engagement in extramarital affairs."
        )
    else:
        response = "No"
        strength = 60
        confidence = 60
        explanation_parts.append(
            "After adjustment, the association between having children and engagement in extramarital affairs is not clearly protective or statistically robust, "
            "so the data do not convincingly support the claim that children decrease affairs."
        )

    explanation_parts.append(
        "However, these results are observational and based on self-reported survey data from a specific population in the late 1960s, "
        "so while they support an association, they do not prove that having children causally reduces extramarital affairs."
    )

    explanation = " ".join(explanation_parts)

    conclusion = {
        "response": response,
        "strength": strength,
        "confidence": confidence,
        "explanation": explanation,
    }

    with open("conclusion.txt", "w", encoding="utf-8") as f:
        json.dump(conclusion, f, ensure_ascii=False)


if __name__ == "__main__":
    main()

