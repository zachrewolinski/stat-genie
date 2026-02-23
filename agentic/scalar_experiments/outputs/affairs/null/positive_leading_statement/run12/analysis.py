import json
from typing import Dict

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def main() -> None:
    # Load metadata (for context) and dataset
    with open("info.json", "r") as f:
        _info = json.load(f)

    df = pd.read_csv("affairs.csv")

    # Binary outcome: any extramarital affair in the past year
    df = df.copy()
    df["had_affair"] = (df["affairs"] > 0).astype(int)
    df["children_num"] = (df["children"] == "yes").astype(int)

    # Descriptive statistics by children status
    group_stats = df.groupby("children")["had_affair"].agg(["mean", "sum", "count"])
    props: Dict[str, float] = group_stats["mean"].to_dict()
    counts_pos: Dict[str, float] = group_stats["sum"].to_dict()
    counts_total: Dict[str, float] = group_stats["count"].to_dict()

    # Unadjusted logistic regression: children only
    model_unadj = smf.logit("had_affair ~ children_num", data=df).fit(disp=0)
    coef_unadj = float(model_unadj.params["children_num"])
    pval_unadj = float(model_unadj.pvalues["children_num"])

    # Adjusted logistic regression including key covariates
    model_adj = smf.logit(
        "had_affair ~ children_num + age + yearsmarried + religiousness + "
        "education + occupation + rating + C(gender)",
        data=df,
    ).fit(disp=0)
    coef_adj = float(model_adj.params["children_num"])
    pval_adj = float(model_adj.pvalues["children_num"])

    # Predicted probabilities at typical covariate values (means and modal gender)
    mean_covs = df[["age", "yearsmarried", "religiousness", "education", "occupation", "rating"]].mean()
    gender_mode = df["gender"].mode().iloc[0]

    pred_df = pd.DataFrame(
        {
            "children_num": [0, 1],
            "age": [mean_covs["age"]] * 2,
            "yearsmarried": [mean_covs["yearsmarried"]] * 2,
            "religiousness": [mean_covs["religiousness"]] * 2,
            "education": [mean_covs["education"]] * 2,
            "occupation": [mean_covs["occupation"]] * 2,
            "rating": [mean_covs["rating"]] * 2,
            "gender": [gender_mode] * 2,
        }
    )

    pred_probs = model_adj.predict(pred_df)
    prob_no_child = float(pred_probs.iloc[0])
    prob_child = float(pred_probs.iloc[1])
    delta_prob = prob_child - prob_no_child

    # Determine direction and statistical significance from the adjusted model
    alpha = 0.05
    significant = pval_adj < alpha
    direction = "decrease" if coef_adj < 0 else "increase"
    abs_delta = abs(delta_prob)

    # Map evidence to a 0–100 Likert-style scale
    if not significant:
        answer_label = "No"
        if abs_delta < 0.02:
            response = 10
        elif abs_delta < 0.05:
            response = 20
        else:
            response = 30
    else:
        if direction == "decrease":
            answer_label = "Yes"
            if abs_delta >= 0.15:
                response = 85
            elif abs_delta >= 0.08:
                response = 75
            elif abs_delta >= 0.04:
                response = 65
            else:
                response = 55
        else:
            answer_label = "No"
            if abs_delta >= 0.15:
                response = 15
            elif abs_delta >= 0.08:
                response = 25
            else:
                response = 35

    # Effect sizes for explanation
    or_unadj = float(np.exp(coef_unadj))
    or_adj = float(np.exp(coef_adj))

    prop_yes = float(props.get("yes", np.nan)) * 100.0
    prop_no = float(props.get("no", np.nan)) * 100.0
    count_yes_affair = int(counts_pos.get("yes", 0))
    count_yes_total = int(counts_total.get("yes", 0))
    count_no_affair = int(counts_pos.get("no", 0))
    count_no_total = int(counts_total.get("no", 0))

    explanation_lines = []
    explanation_lines.append(
        "Research question: Does having children decrease engagement in extramarital affairs?"
    )
    explanation_lines.append(
        f"In the sample of {len(df)} married individuals, {count_yes_affair} out of {count_yes_total} "
        f"people with children ({prop_yes:.1f}%) reported at least one affair in the past year, "
        f"compared with {count_no_affair} out of {count_no_total} people without children ({prop_no:.1f}%)."
    )
    explanation_lines.append(
        "I modeled the probability of having at least one affair (binary outcome) using logistic regression."
    )
    explanation_lines.append(
        f"An unadjusted model with only a children indicator gave an odds ratio of {or_unadj:.2f} "
        f"for people with children relative to those without (p = {pval_unadj:.3g})."
    )
    explanation_lines.append(
        "To account for potential confounding, I then fit an adjusted logistic model including age, "
        "years married, religiousness, education, occupation, self-rated marital happiness, and gender."
    )
    explanation_lines.append(
        f"In the adjusted model, the odds ratio associated with having children was {or_adj:.2f} "
        f"with p = {pval_adj:.3g}."
    )
    explanation_lines.append(
        "Using the adjusted model, I computed predicted probabilities at typical covariate values. "
        f"The predicted probability of having an affair was {prob_no_child:.1%} for someone without children "
        f"and {prob_child:.1%} for someone with children, a difference of {delta_prob:+.1%}."
    )
    if not significant:
        explanation_lines.append(
            "Because the adjusted association between having children and the likelihood of an affair "
            "is not statistically significant at the 5% level and the estimated effect size is modest, "
            "the data do not provide strong evidence that having children changes engagement in extramarital affairs."
        )
    else:
        if direction == "decrease":
            explanation_lines.append(
                "Because the adjusted association is statistically significant and the estimated effect "
                "indicates that having children is associated with a lower probability of affairs, "
                "there is evidence that having children is linked to reduced engagement in extramarital affairs."
            )
        else:
            explanation_lines.append(
                "Because the adjusted association is statistically significant but the estimated effect "
                "indicates that having children is associated with a higher probability of affairs, "
                "the data suggest that having children does not decrease engagement in extramarital affairs "
                "and may even be associated with a slight increase."
            )
    explanation_lines.append(
        "On a 0–100 scale where higher values correspond to a stronger 'Yes' answer to the research question, "
        f"I summarize this evidence with a score of {response} out of 100, corresponding to a '{answer_label}' "
        "answer given the direction, statistical significance, and magnitude of the estimated effect."
    )

    explanation = " ".join(explanation_lines)

    output = {
        "response": int(response),
        "explanation": explanation,
    }
    with open("conclusion.txt", "w") as f:
        json.dump(output, f)


if __name__ == "__main__":
    main()

