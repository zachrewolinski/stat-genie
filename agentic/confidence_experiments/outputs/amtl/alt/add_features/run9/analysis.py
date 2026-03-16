import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import patsy


def main() -> None:
    data_path = Path("amtl.csv")
    info_path = Path("info.json")

    if not data_path.exists():
        raise FileNotFoundError("amtl.csv not found in working directory")

    df = pd.read_csv(data_path)

    # Basic validity filters for the binomial outcome
    df = df.copy()
    df = df[(df["sockets"] > 0) & (df["num_amtl"] >= 0) & (df["num_amtl"] <= df["sockets"])]

    # Keep only the genera relevant to the research question
    target_genera = ["Homo sapiens", "Pan", "Pongo", "Papio"]
    df = df[df["genus"].isin(target_genera)].copy()

    # Create main predictor: human vs non-human primate
    df["is_human"] = (df["genus"] == "Homo sapiens").astype(int)

    # Drop rows with missing key covariates
    df = df.dropna(subset=["age", "prob_male", "tooth_class", "is_human"])

    # Prepare binomial response: successes = missing teeth, failures = present teeth
    successes = df["num_amtl"].to_numpy()
    failures = (df["sockets"] - df["num_amtl"]).to_numpy()
    endog = np.column_stack([successes, failures])

    # Design matrix with intercept, is_human, age, sex proxy, and tooth class
    design_formula = "is_human + age + prob_male + C(tooth_class)"
    exog = patsy.dmatrix(design_formula, df, return_type="dataframe")
    design_info = exog.design_info

    model = sm.GLM(endog, exog, family=sm.families.Binomial())
    result = model.fit()

    # Extract key effect: human vs non-human
    coef = result.params["is_human"]
    se = result.bse["is_human"]
    pval = result.pvalues["is_human"]
    odds_ratio = float(np.exp(coef))
    ci_low = float(np.exp(coef - 1.96 * se))
    ci_high = float(np.exp(coef + 1.96 * se))

    # Compute illustrative predicted probabilities at typical values
    median_age = float(df["age"].median())
    mean_prob_male = float(df["prob_male"].mean())
    tooth_class_ref = df["tooth_class"].mode().iat[0]

    # Build small design matrices for prediction
    pred_df = pd.DataFrame(
        {
            "is_human": [0, 1],
            "age": [median_age, median_age],
            "prob_male": [mean_prob_male, mean_prob_male],
            "tooth_class": [tooth_class_ref, tooth_class_ref],
        }
    )
    # Use the original design_info to ensure prediction matrix matches training columns
    pred_exog = patsy.build_design_matrices([design_info], pred_df)[0]
    pred_prob = result.predict(pred_exog)
    nonhuman_prob = float(pred_prob[0])
    human_prob = float(pred_prob[1])

    # Map statistical evidence to a 0–100 Likert score
    if pval < 0.05 and odds_ratio > 1.0:
        # Evidence that humans have higher AMTL
        base = 60
        # Scale with both strength of association and significance
        effect_strength = min(max(odds_ratio - 1.0, 0.0), 3.0) / 3.0
        significance_strength = min(max(-np.log10(max(pval, 1e-12)) / 6.0, 0.0), 1.0)
        response_score = base + int(40 * (0.6 * effect_strength + 0.4 * significance_strength))
    elif pval < 0.05 and odds_ratio < 1.0:
        # Evidence that humans have lower AMTL than non-humans
        base = 60
        effect_strength = min(max((1.0 / max(odds_ratio, 1e-12)) - 1.0, 0.0), 3.0) / 3.0
        significance_strength = min(max(-np.log10(max(pval, 1e-12)) / 6.0, 0.0), 1.0)
        response_score = base + int(40 * (0.6 * effect_strength + 0.4 * significance_strength))
        # Interpret lower odds as strong evidence against the research hypothesis,
        # so invert around 50 to keep 0 = strong "No"
        response_score = max(0, 100 - response_score)
    else:
        # No clear statistical evidence either way
        # Start near neutral "No" and let weaker p-values move us closer to 50.
        if pval >= 0.5:
            response_score = 10
        else:
            # Between 0.05 and 0.5: modest suggestion but not conventionally significant
            # Map linearly from 20 (p=0.5) to 40 (p=0.05)
            t = (0.5 - pval) / (0.5 - 0.05)
            response_score = int(20 + 20 * max(0.0, min(1.0, t)))

    response_score = int(max(0, min(100, response_score)))

    # Load research question text for context
    research_question = (
        "Do modern humans (Homo sapiens) have higher frequencies of "
        "antemortem tooth loss (AMTL) compared to non-human primate genera "
        "(Pan, Pongo, Papio), after accounting for the effects of age, sex, "
        "and tooth class?"
    )
    if info_path.exists():
        try:
            with info_path.open() as f:
                info = json.load(f)
            if isinstance(info, dict) and "research_questions" in info and info["research_questions"]:
                research_question = str(info["research_questions"][0])
        except Exception:
            # Fall back to default text if info.json is malformed
            pass

    n_rows = int(df.shape[0])
    n_specimens = int(df["specimen"].nunique())
    genus_counts = df["genus"].value_counts().to_dict()

    direction = "higher" if odds_ratio > 1.0 else "lower"
    sig_text = "statistically significant" if pval < 0.05 else "not statistically significant at the 0.05 level"

    explanation_parts = [
        f"Research question: {research_question}",
        "",
        "Data and model:",
        f"- Analyzed {n_rows} tooth-class observations from {n_specimens} specimens across genera {list(genus_counts.keys())}.",
        "- Modeled the number of missing teeth (num_amtl) out of observable sockets using a binomial logistic regression.",
        "- Included predictors for human vs non-human (is_human), age at death (age), probability of being male (prob_male) as a sex proxy, and tooth class (tooth_class).",
        "",
        "Key results for human vs non-human:",
        f"- Estimated odds ratio for AMTL in humans vs non-human primates (controlling for age, sex, and tooth class): {odds_ratio:.2f}.",
        f"- 95% confidence interval for this odds ratio: [{ci_low:.2f}, {ci_high:.2f}].",
        f"- p-value for the human vs non-human effect: {pval:.3g} ({sig_text}).",
        f"- At median age {median_age:.1f} years, mean sex probability, and the most common tooth class ({tooth_class_ref}), "
        f"the model predicts an AMTL probability of {nonhuman_prob:.3f} for non-human primates and {human_prob:.3f} for humans, "
        f"indicating {direction} AMTL in humans on this scale.",
        "",
        "Conclusion:",
    ]

    if pval < 0.05 and odds_ratio > 1.0:
        conclusion_sentence = (
            "There is statistically significant evidence that modern humans have higher frequencies of "
            "antemortem tooth loss than the non-human primate genera considered, even after accounting "
            "for age, sex, and tooth class."
        )
    elif pval < 0.05 and odds_ratio < 1.0:
        conclusion_sentence = (
            "There is statistically significant evidence that modern humans actually have lower frequencies "
            "of antemortem tooth loss than the non-human primate genera considered, after accounting for "
            "age, sex, and tooth class, which runs counter to the original hypothesis."
        )
    else:
        conclusion_sentence = (
            "The analysis does not provide statistically significant evidence that modern humans differ from "
            "the non-human primate genera in their frequencies of antemortem tooth loss once age, sex, and "
            "tooth class are taken into account; any apparent differences could be due to sampling variability."
        )

    explanation_parts.append(conclusion_sentence)

    explanation = "\n".join(explanation_parts)

    conclusion = {
        "response": response_score,
        "explanation": explanation,
    }

    with Path("conclusion.txt").open("w") as f:
        json.dump(conclusion, f)

    # Also print a brief summary to stdout for interactive inspection
    print(f"response={response_score}")
    print(conclusion_sentence)


if __name__ == "__main__":
    main()
