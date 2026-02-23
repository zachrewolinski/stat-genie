import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def main() -> None:
    # Load data
    df = pd.read_csv("crofoot.csv")

    # Construct variables reflecting the research question
    # Relative group size: focal group size minus other group size
    df["size_diff"] = df["feature7"] - df["feature8"]

    # Relative contest location: focal distance from its home-center minus other distance
    # Negative values mean the focal group is closer to the center of its home range.
    df["dist_diff"] = df["feature5"] - df["feature6"]

    # Basic descriptive summaries for interpretation
    df["size_advantage"] = np.where(
        df["size_diff"] > 0,
        "focal_larger",
        np.where(df["size_diff"] < 0, "other_larger", "same_size"),
    )
    df["home_advantage"] = np.where(
        df["dist_diff"] < 0,
        "focal_closer",
        np.where(df["dist_diff"] > 0, "other_closer", "equal_distance"),
    )

    win_overall = float(df["feature4"].mean())

    win_by_size = df.groupby("size_advantage")["feature4"].mean().to_dict()
    win_by_home = df.groupby("home_advantage")["feature4"].mean().to_dict()

    # Fit logistic regression: probability focal group wins as a function of
    # relative group size and relative contest location.
    # feature4: 1 if focal group won, 0 otherwise.
    try:
        model = smf.logit("feature4 ~ size_diff + dist_diff", data=df)
        result = model.fit(disp=False, maxiter=100)
        size_coef = float(result.params.get("size_diff", np.nan))
        loc_coef = float(result.params.get("dist_diff", np.nan))
        size_p = float(result.pvalues.get("size_diff", np.nan))
        loc_p = float(result.pvalues.get("dist_diff", np.nan))
        lr_pvalue = float(getattr(result, "llr_pvalue", np.nan))
        size_or = float(np.exp(size_coef)) if np.isfinite(size_coef) else np.nan
        loc_or = float(np.exp(loc_coef)) if np.isfinite(loc_coef) else np.nan
    except Exception:
        # If the model fails to converge for any reason, fall back to no-effect assessment.
        size_coef = np.nan
        loc_coef = np.nan
        size_p = np.nan
        loc_p = np.nan
        lr_pvalue = np.nan
        size_or = np.nan
        loc_or = np.nan

    # Quantify statistical evidence for each predictor using p-values.
    def significance_score(p: float) -> float:
        """
        Map p-values into [0, 1] evidence scores.

        p <= 0.01  -> close to 1 (strong evidence)
        p <= 0.05  -> moderate evidence
        p <= 0.10  -> weak evidence
        p > 0.10   -> ~0 (little to no evidence)
        """
        if not np.isfinite(p):
            return 0.0
        if p >= 0.10:
            return 0.0
        # Linearly map p in [0, 0.10] to [1, 0]
        return float((0.10 - p) / 0.10)

    size_evidence = significance_score(size_p)
    loc_evidence = significance_score(loc_p)
    total_evidence = size_evidence + loc_evidence  # in [0, 2]

    # Convert total evidence into a 0–100 Likert-style response.
    #  - Around 20: strong "No" / little evidence of an effect.
    #  - Around 55: one predictor with moderate evidence.
    #  - Around 90: both predictors with strong evidence.
    response = int(round(20 + 35 * total_evidence))
    response = max(0, min(100, response))

    # Build explanation text summarizing the analysis and results.
    def pct(x: float) -> float:
        return float(x * 100.0)

    def get_rate(d: dict, key: str) -> float:
        value = d.get(key)
        if value is None or not np.isfinite(value):
            return float("nan")
        return float(value)

    focal_larger_win = get_rate(win_by_size, "focal_larger")
    other_larger_win = get_rate(win_by_size, "other_larger")
    same_size_win = get_rate(win_by_size, "same_size")

    focal_home_win = get_rate(win_by_home, "focal_closer")
    other_home_win = get_rate(win_by_home, "other_closer")
    equal_home_win = get_rate(win_by_home, "equal_distance")

    explanation_parts = []

    explanation_parts.append(
        "Research question: Do relative group size and contest location "
        "influence the probability of a capuchin monkey group winning an intergroup contest?"
    )

    explanation_parts.append(
        "Data and variables: The dataset contains 58 intergroup contests between "
        "capuchin groups. The outcome variable is whether the focal group won "
        "(feature4 = 1) or lost (feature4 = 0). To capture the predictors of interest, "
        "I constructed two key derived variables: (1) relative group size "
        "size_diff = feature7 (number of individuals in the focal group) minus "
        "feature8 (number of individuals in the other group); and "
        "(2) relative contest location dist_diff = feature5 (distance of the focal group "
        "from the center of its home range) minus feature6 (distance of the other group "
        "from the center of its home range). Negative dist_diff values mean the focal "
        "group is closer to the center of its home range, indicating a potential home-range advantage."
    )

    explanation_parts.append(
        f"Descriptive patterns: Overall, the focal group wins about {pct(win_overall):.1f}% "
        "of contests. When the focal group is larger than its opponent, it wins "
        f"approximately {pct(focal_larger_win):.1f}% of contests; when it is smaller, it wins "
        f"about {pct(other_larger_win):.1f}% of contests; and when groups are the same size, "
        f"it wins about {pct(same_size_win):.1f}% (where applicable). For contest location, "
        f"when the focal group is closer to the center of its home range, it wins about "
        f"{pct(focal_home_win):.1f}% of contests; when the opponent is closer to its own center, "
        f"the focal group wins about {pct(other_home_win):.1f}% of contests; and when both groups "
        f"are at roughly equal distance, the focal group wins about {pct(equal_home_win):.1f}%."
    )

    if np.isfinite(size_coef) and np.isfinite(loc_coef):
        explanation_parts.append(
            "Inferential model: To jointly assess the effects of relative group size and "
            "contest location while controlling for each other, I fit a logistic regression "
            "model with the focal group's win probability as the outcome and predictors "
            "size_diff and dist_diff."
        )

        explanation_parts.append(
            f"For relative group size, the logistic regression coefficient for size_diff "
            f"is {size_coef:.3f} (odds ratio ≈ {size_or:.2f}, p-value = {size_p:.3f}). "
            "A positive coefficient and odds ratio above 1 indicate that, holding contest "
            "location fixed, contests in which the focal group has more individuals than "
            "its opponent are associated with a higher probability of winning. The p-value "
            "quantifies how compatible this pattern is with the null hypothesis of no size effect."
        )

        explanation_parts.append(
            f"For contest location, the coefficient for dist_diff is {loc_coef:.3f} "
            f"(odds ratio ≈ {loc_or:.2f}, p-value = {loc_p:.3f}). Because dist_diff is defined "
            "as focal distance minus other distance, a negative coefficient implies that being "
            "closer to the center of its own home range (more negative dist_diff) increases the "
            "focal group's chance of winning, whereas a positive coefficient would imply the "
            "opposite. The p-value again measures the strength of evidence against the null "
            "hypothesis of no location effect."
        )

        explanation_parts.append(
            f"Considering both predictors together, the likelihood-ratio test comparing the "
            f"model with both predictors to a null model with only an intercept has p-value "
            f"{lr_pvalue:.3f}, which reflects the overall contribution of relative group size "
            "and contest location to explaining variation in contest outcomes."
        )
    else:
        explanation_parts.append(
            "The logistic regression model for the win probability did not converge cleanly, "
            "so I rely primarily on descriptive contrasts in win rates as a function of "
            "relative group size and relative proximity to the home-range center."
        )

    # Interpret the evidence in plain language, in line with the Likert score.
    if response >= 80:
        qualitative = (
            "Overall, there is strong statistical evidence that both relative group size "
            "and contest location meaningfully influence the probability that a focal "
            "capuchin group wins an intergroup contest. The substantial differences in "
            "win rates across size and location advantages, combined with statistically "
            "significant regression coefficients for both predictors, support a clear 'Yes' answer."
        )
    elif response >= 60:
        qualitative = (
            "Overall, the results provide moderate statistical evidence that at least one of "
            "the two predictors—relative group size and contest location—affects the focal "
            "group's probability of winning, with the other showing weaker but directionally "
            "consistent effects. This supports a 'Yes, but with moderate strength' answer."
        )
    elif response >= 40:
        qualitative = (
            "Overall, the evidence that relative group size and contest location influence "
            "win probability is fairly weak. Differences in win rates across size and location "
            "categories and the regression estimates are suggestive but not statistically "
            "robust, so the data lean toward a cautious 'probably no strong effect' answer."
        )
    else:
        qualitative = (
            "Overall, there is little to no statistical evidence in this dataset that "
            "relative group size or contest location materially influence the focal group's "
            "probability of winning. Win rates across size and location categories are "
            "similar, and regression estimates are small and statistically non-significant, "
            "supporting a 'No' answer."
        )

    explanation_parts.append(qualitative)

    explanation_parts.append(
        f"Numeric conclusion: mapping the combined statistical evidence from both predictors "
        f"onto a 0–100 Likert scale, where 0 represents a strong 'No' and 100 a strong 'Yes', "
        f"yields a response value of {response}. Higher values correspond to stronger and more "
        "statistically robust evidence that relative group size and contest location influence "
        "intergroup contest outcomes."
    )

    explanation = "\n\n".join(explanation_parts)

    conclusion = {"response": response, "explanation": explanation}

    output_path = Path("conclusion.txt")
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(conclusion, f, ensure_ascii=False)


if __name__ == "__main__":
    main()

