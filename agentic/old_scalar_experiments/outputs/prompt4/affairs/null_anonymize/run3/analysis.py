import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def main() -> None:
    data_path = Path("affairs.csv")
    df = pd.read_csv(data_path)

    # Outcome: any extramarital intercourse in past year
    df["had_affair"] = (df["feature2"] > 0).astype(int)

    # Key predictor: children in the marriage (1 = yes, 0 = no)
    df["children_yes"] = (df["feature6"].str.lower() == "yes").astype(int)

    # Basic descriptive statistics by children status
    grouped = df.groupby("children_yes")
    counts = grouped["had_affair"].size()
    affair_rate = grouped["had_affair"].mean()
    affair_freq_mean = grouped["feature2"].mean()

    # Logistic regression controlling for other factors
    X = pd.DataFrame(
        {
            "children_yes": df["children_yes"],
            # Gender: 1 = male, 0 = female
            "male": (df["feature3"].str.lower() == "male").astype(int),
            "age": df["feature4"],
            "years_married": df["feature5"],
            "religiousness": df["feature7"],
            "education": df["feature8"],
            "occupation": df["feature9"],
            "marriage_rating": df["feature10"],
        }
    )
    X = sm.add_constant(X, has_constant="add")
    y = df["had_affair"]

    logit_model = sm.Logit(y, X)
    logit_res = logit_model.fit(disp=False)

    coef_children = float(logit_res.params["children_yes"])
    p_children = float(logit_res.pvalues["children_yes"])
    conf_int = logit_res.conf_int().loc["children_yes"]
    coef_ci_low = float(conf_int[0])
    coef_ci_high = float(conf_int[1])
    odds_ratio = float(np.exp(coef_children))
    or_ci_low = float(np.exp(coef_ci_low))
    or_ci_high = float(np.exp(coef_ci_high))

    # Map evidence to a 0-100 "Yes" scale
    # We interpret a negative, statistically significant coefficient together with
    # lower descriptive rates as evidence that having children decreases affairs.
    children_has_lower_rate = affair_rate.loc[1] < affair_rate.loc[0]
    strong_significance = p_children < 0.01
    moderate_significance = 0.01 <= p_children < 0.05

    if children_has_lower_rate and strong_significance and odds_ratio < 1.0:
        response_score = 85
    elif children_has_lower_rate and (strong_significance or moderate_significance) and odds_ratio < 1.0:
        response_score = 70
    elif children_has_lower_rate and odds_ratio < 1.0:
        response_score = 60
    elif odds_ratio < 1.0:
        response_score = 55
    elif odds_ratio > 1.0 and strong_significance:
        response_score = 20
    elif odds_ratio > 1.0:
        response_score = 35
    else:
        response_score = 50

    # Build textual explanation
    explanation_parts = []

    explanation_parts.append(
        "Research question: Does having children decrease engagement in extramarital affairs?"
    )

    # Descriptive summary
    rate_children = affair_rate.loc[1]
    rate_no_children = affair_rate.loc[0]
    mean_freq_children = affair_freq_mean.loc[1]
    mean_freq_no_children = affair_freq_mean.loc[0]

    explanation_parts.append(
        f"Descriptively, among individuals with children (n={counts.loc[1]}), "
        f"{rate_children:.3f} had at least one extramarital affair in the past year, "
        f"with an average affair-frequency score of {mean_freq_children:.3f}."
    )
    explanation_parts.append(
        f"Among individuals without children (n={counts.loc[0]}), "
        f"{rate_no_children:.3f} had at least one affair, "
        f"with an average affair-frequency score of {mean_freq_no_children:.3f}."
    )

    # Regression summary
    direction = "decreases" if coef_children < 0 else "increases"
    explanation_parts.append(
        "To adjust for potential confounders (gender, age, years married, religiousness, "
        "education, occupation, and self-rated marriage quality), I fit a logistic "
        "regression predicting whether a respondent had any affair."
    )
    explanation_parts.append(
        f"In this model, the coefficient for having children is {coef_children:.3f} "
        f"(95% CI {coef_ci_low:.3f} to {coef_ci_high:.3f}, p={p_children:.3g}), "
        f"corresponding to an odds ratio of {odds_ratio:.3f} "
        f"(95% CI {or_ci_low:.3f} to {or_ci_high:.3f}). "
        f"This indicates that, holding the other variables constant, having children "
        f"{direction} the odds of engaging in extramarital affairs."
    )

    # Overall conclusion
    if response_score > 55:
        overall_conclusion = (
            "Overall, both the descriptive statistics and the regression results suggest "
            "that having children is associated with a lower level of engagement in "
            "extramarital affairs, although this is still an observational association "
            "rather than definitive causal evidence."
        )
    elif response_score < 45:
        overall_conclusion = (
            "Overall, the evidence suggests that having children is associated with a "
            "higher level of engagement in extramarital affairs, although causality "
            "cannot be established from this observational dataset."
        )
    else:
        overall_conclusion = (
            "Overall, the evidence does not clearly show that having children reduces "
            "engagement in extramarital affairs; the association is weak or statistically "
            "uncertain in this dataset."
        )

    explanation_parts.append(overall_conclusion)
    explanation_parts.append(
        f"On a 0–100 scale where higher scores indicate stronger evidence that "
        f"having children decreases engagement in extramarital affairs, I assign a "
        f"score of {response_score} based on this analysis."
    )

    explanation = " ".join(explanation_parts)

    conclusion = {"response": int(response_score), "explanation": explanation}

    with open("conclusion.txt", "w", encoding="utf-8") as f:
        json.dump(conclusion, f, ensure_ascii=False)


if __name__ == "__main__":
    main()

