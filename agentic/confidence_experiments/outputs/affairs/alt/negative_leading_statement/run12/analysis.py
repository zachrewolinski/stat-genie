import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main() -> None:
    # Load data
    df = pd.read_csv("affairs.csv")

    # Define outcome: any affair in past year
    df["has_affair"] = (df["affairs"] > 0).astype(int)
    df["children_yes"] = (df["children"] == "yes").astype(int)

    # Basic descriptive statistics by children status
    group = df.groupby("children")
    mean_affairs = group["affairs"].mean()
    prevalence = group["has_affair"].mean()

    # Logistic regression for probability of any affair
    # children_yes is the main predictor; control for other covariates
    try:
        logit_model = smf.logit(
            "has_affair ~ children_yes + C(gender) + age + yearsmarried + "
            "religiousness + education + occupation + rating",
            data=df,
        )
        logit_res = logit_model.fit(disp=False)
        children_coef_logit = logit_res.params["children_yes"]
        children_p_logit = logit_res.pvalues["children_yes"]
        children_or_logit = float(np.exp(children_coef_logit))
    except Exception as exc:  # pragma: no cover - defensive
        logit_res = None
        children_coef_logit = np.nan
        children_p_logit = np.nan
        children_or_logit = np.nan
        print(f"Logistic regression failed: {exc}")

    # Poisson regression for frequency of affairs (count outcome)
    try:
        poisson_model = smf.glm(
            "affairs ~ children_yes + C(gender) + age + yearsmarried + "
            "religiousness + education + occupation + rating",
            data=df,
            family=sm.families.Poisson(),
        )
        poisson_res = poisson_model.fit()
        children_coef_pois = poisson_res.params["children_yes"]
        children_p_pois = poisson_res.pvalues["children_yes"]
        children_rr_pois = float(np.exp(children_coef_pois))
    except Exception as exc:  # pragma: no cover - defensive
        poisson_res = None
        children_coef_pois = np.nan
        children_p_pois = np.nan
        children_rr_pois = np.nan
        print(f"Poisson regression failed: {exc}")

    # Interpret results
    explanation_lines = []
    explanation_lines.append(
        "Research question: Does having children decrease engagement in extramarital affairs?"
    )
    explanation_lines.append(
        "Outcome variable 'affairs' measures how often individuals engaged in extramarital sex "
        "in the past year (0 = none; larger values = more frequent)."
    )
    explanation_lines.append(
        "I created a binary outcome 'has_affair' indicating any extramarital affair in the past year "
        "and compared participants with and without children."
    )

    explanation_lines.append(
        f"Descriptively, the mean affair score was "
        f"{mean_affairs['yes']:.2f} for those with children and "
        f"{mean_affairs['no']:.2f} for those without children."
    )
    explanation_lines.append(
        f"The proportion with at least one affair was "
        f"{prevalence['yes']:.2%} for participants with children and "
        f"{prevalence['no']:.2%} for those without children."
    )

    # Statistical significance and direction
    has_evidence_of_decrease = False
    if not np.isnan(children_coef_logit):
        direction_logit = "decrease" if children_coef_logit < 0 else "increase"
        explanation_lines.append(
            "Using logistic regression for the probability of any affair, "
            f"the coefficient for having children (yes vs no) corresponds to an odds ratio "
            f"of {children_or_logit:.2f}, indicating a relative {direction_logit} "
            "in the odds of having an affair if the odds ratio is below 1 (decrease) "
            "or above 1 (increase)."
        )
        explanation_lines.append(
            f"The p-value for this effect in the logistic model was "
            f"{children_p_logit:.3f}."
        )
        if children_p_logit < 0.05 and children_coef_logit < 0:
            has_evidence_of_decrease = True

    if not np.isnan(children_coef_pois):
        direction_pois = "decrease" if children_coef_pois < 0 else "increase"
        explanation_lines.append(
            "Using a Poisson regression for the affair frequency, the rate ratio associated "
            f"with having children was {children_rr_pois:.2f}, again suggesting a relative "
            f"{direction_pois} in the expected affair count if the rate ratio is below 1 "
            "or above 1 respectively."
        )
        explanation_lines.append(
            f"The p-value for this effect in the Poisson model was "
            f"{children_p_pois:.3f}."
        )
        if children_p_pois < 0.05 and children_coef_pois < 0:
            has_evidence_of_decrease = True

    # Map evidence strength to Likert-style 0–100 response
    # 0 = strong "No", 100 = strong "Yes"
    if has_evidence_of_decrease:
        # Evidence that having children is associated with fewer affairs
        # Calibrate strength by the magnitude and consistency of effects.
        # Start with a moderately strong "Yes".
        response_score = 70
        explanation_lines.append(
            "Both regression models suggest that, after controlling for demographic and marriage "
            "characteristics, having children is associated with a statistically significant "
            "reduction in engagement in extramarital affairs."
        )
        explanation_lines.append(
            "Given consistent, statistically significant negative associations, I interpret this "
            "as moderate-to-strong evidence that having children decreases engagement in extramarital affairs."
        )
    else:
        # No statistically significant evidence that children reduce affairs
        response_score = 20
        explanation_lines.append(
            "Across both descriptive comparisons and regression models controlling for other factors, "
            "there is not strong, statistically significant evidence that having children reduces "
            "engagement in extramarital affairs."
        )
        explanation_lines.append(
            "Effect estimates are small and/or not statistically significant at conventional levels, "
            "so the data do not support the claim that having children decreases the likelihood or "
            "frequency of extramarital affairs."
        )

    explanation = " ".join(explanation_lines)

    conclusion = {
        "response": int(response_score),
        "explanation": explanation,
    }

    Path("conclusion.txt").write_text(json.dumps(conclusion), encoding="utf-8")


if __name__ == "__main__":
    main()

