import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def main() -> None:
    base_dir = Path(__file__).parent

    # Load metadata and data
    info_path = base_dir / "info.json"
    data_path = base_dir / "affairs.csv"

    with info_path.open("r", encoding="utf-8") as f:
        info = json.load(f)

    df = pd.read_csv(data_path)

    # Prepare variables
    df["has_children"] = df["feature6"].map({"yes": 1, "no": 0})
    df["any_affair"] = (df["feature2"] > 0).astype(int)

    # Basic sanity: drop rows with missing key fields if any
    df = df.dropna(
        subset=[
            "has_children",
            "any_affair",
            "feature3",
            "feature4",
            "feature5",
            "feature7",
            "feature8",
            "feature9",
            "feature10",
        ]
    )

    n_obs = int(df.shape[0])

    # Descriptive statistics by children status
    affair_rate = df.groupby("has_children")["any_affair"].agg(["mean", "sum", "count"])
    affair_intensity = df.groupby("has_children")["feature2"].mean()

    # Map for readability
    rate_with_children = float(affair_rate.loc[1, "mean"])
    rate_without_children = float(affair_rate.loc[0, "mean"])
    mean_affairs_with_children = float(affair_intensity.loc[1])
    mean_affairs_without_children = float(affair_intensity.loc[0])

    # Logistic regression: probability of any affair ~ children + controls
    # feature3: gender (categorical)
    # feature4: age (numeric code)
    # feature5: years married
    # feature7: religiousness
    # feature8: education
    # feature9: occupation
    # feature10: marriage rating
    formula = (
        "any_affair ~ has_children + C(feature3) + feature4 + feature5 + "
        "feature7 + feature8 + feature9 + feature10"
    )

    try:
        logit_model = smf.logit(formula=formula, data=df).fit(disp=False)
        coef_children = float(logit_model.params["has_children"])
        pval_children = float(logit_model.pvalues["has_children"])
        # Convert log-odds coefficient to odds ratio
        odds_ratio_children = float(np.exp(coef_children))
    except Exception:
        # If the model fails for any reason, fall back to a simpler comparison
        logit_model = None
        coef_children = np.nan
        pval_children = np.nan
        odds_ratio_children = np.nan

    # Decide response, strength, and confidence
    # Interpretation logic:
    # - If coefficient is negative and statistically significant, children are associated
    #   with a lower chance of affairs -> "Yes".
    # - If coefficient is positive and significant, or clearly positive descriptively,
    #   then having children does not decrease affairs -> "No".
    # - If results are inconclusive, answer "No" with low strength and confidence.

    if not np.isnan(coef_children) and not np.isnan(pval_children):
        if coef_children < 0 and pval_children < 0.05:
            response = "Yes"
            # Strong evidence of a decrease
            strength = 80
            confidence = 80
        elif coef_children > 0 and pval_children < 0.05:
            response = "No"
            # Strong evidence in the opposite direction
            strength = 85
            confidence = 85
        else:
            # Weak or non-significant effect
            response = "No"
            strength = 30
            confidence = 40
    else:
        # Model failure: rely purely on descriptive differences
        if rate_with_children < rate_without_children and mean_affairs_with_children < mean_affairs_without_children:
            response = "Yes"
            strength = 50
            confidence = 50
        else:
            response = "No"
            strength = 40
            confidence = 40

    # Build explanation string
    question = info.get("research_questions", [""])[0]

    explanation_parts = []
    explanation_parts.append(
        f"Research question: '{question.strip()}' (n = {n_obs} married individuals)."
    )
    explanation_parts.append(
        "Extramarital engagement was defined as reporting any non-zero frequency of extramarital sexual intercourse "
        "in the past year."
    )
    explanation_parts.append(
        f"Among those without children, {rate_without_children:.3f} of respondents reported at least one extramarital affair, "
        f"with an average affair frequency of {mean_affairs_without_children:.3f} on the 0–12 scale."
    )
    explanation_parts.append(
        f"Among those with children, {rate_with_children:.3f} reported at least one affair, "
        f"with an average affair frequency of {mean_affairs_with_children:.3f}."
    )

    if logit_model is not None and not np.isnan(coef_children):
        direction = "lower" if coef_children < 0 else "higher"
        explanation_parts.append(
            "A logistic regression of having any affair on children status, controlling for gender, age, "
            "years married, religiousness, education, occupation, and marital satisfaction, "
            f"estimated that respondents with children have {direction} odds of reporting an affair "
            f"(odds ratio ≈ {odds_ratio_children:.3f}, p-value ≈ {pval_children:.3f})."
        )
        if pval_children < 0.05:
            explanation_parts.append(
                "Because this association is statistically significant at conventional levels (p < 0.05), "
                "it is unlikely to be due to random sampling variability alone."
            )
        else:
            explanation_parts.append(
                "However, this association is not statistically significant at the 5% level, so the data do not "
                "provide strong evidence that children meaningfully change the likelihood of extramarital affairs."
            )
    else:
        explanation_parts.append(
            "A multivariable logistic regression could not be reliably estimated, so the conclusion is based on "
            "descriptive comparisons of affair prevalence and frequency by children status."
        )

    if response == "Yes":
        explanation_parts.append(
            "Overall, the balance of evidence suggests that having children is associated with a decrease in "
            "engagement in extramarital affairs, although the estimated effect size and its statistical "
            "significance are taken into account in the reported strength and confidence scores."
        )
    else:
        explanation_parts.append(
            "Overall, the data do not support the claim that having children decreases engagement in extramarital "
            "affairs; either the estimated effect is small, not statistically distinguishable from zero, or points "
            "in the opposite direction."
        )

    explanation = " ".join(explanation_parts)

    conclusion = {
        "response": response,
        "strength": int(strength),
        "confidence": int(confidence),
        "explanation": explanation,
    }

    conclusion_path = base_dir / "conclusion.txt"
    with conclusion_path.open("w", encoding="utf-8") as f:
        json.dump(conclusion, f, ensure_ascii=False)


if __name__ == "__main__":
    main()
