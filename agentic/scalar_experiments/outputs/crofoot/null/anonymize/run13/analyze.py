import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def main() -> None:
    base_dir = Path(__file__).parent
    data_path = base_dir / "crofoot.csv"
    df = pd.read_csv(data_path)

    # Outcome: 1 if focal group won, 0 otherwise
    y = df["feature4"]

    # Relative group size: focal size minus other size
    df["size_diff"] = df["feature7"] - df["feature8"]

    # Contest location advantage:
    # distance from home range center for focal (feature5) and other (feature6).
    # Smaller distance => closer to home range center.
    # Define a continuous advantage variable where positive values mean focal is
    # closer to its home range center than the other group is to its own.
    df["loc_adv"] = df["feature6"] - df["feature5"]

    X = df[["size_diff", "loc_adv"]]
    X = sm.add_constant(X)

    model = sm.Logit(y, X)
    result = model.fit(disp=False)

    params = result.params
    pvalues = result.pvalues
    conf_int = result.conf_int()

    # Odds ratios for interpretation
    odds_ratios = np.exp(params)

    size_p = float(pvalues["size_diff"])
    loc_p = float(pvalues["loc_adv"])

    # Basic interpretation flags using a 0.05 significance threshold.
    size_sig = size_p < 0.05
    loc_sig = loc_p < 0.05

    # Map evidence to a 0–100 Likert-style response.
    # We treat "Yes" as evidence that at least one of the predictors
    # (relative group size or contest location) significantly influences
    # win probability.
    if size_sig and loc_sig:
        response = 90
    elif size_sig or loc_sig:
        response = 75
    else:
        response = 25

    explanation_parts = []

    explanation_parts.append(
        "I modeled the probability that the focal capuchin group won an intergroup contest "
        "using logistic regression with two predictors: (1) relative group size "
        "defined as the difference in group size between the focal and other group "
        "(feature7 − feature8), and (2) contest location advantage defined as the "
        "difference in distance from each group to the center of its home range "
        "(feature6 − feature5), where positive values mean the focal group was closer "
        "to its own home range center than the opposing group was to its own."
    )

    explanation_parts.append(
        "The logistic regression estimated a positive/negative coefficient and associated "
        "p-value for each predictor. I examined the coefficient estimates, their odds ratios, "
        "and p-values to determine whether relative group size and contest location were "
        "statistically significant predictors of winning."
    )

    size_coef = float(params["size_diff"])
    loc_coef = float(params["loc_adv"])
    size_or = float(odds_ratios["size_diff"])
    loc_or = float(odds_ratios["loc_adv"])

    explanation_parts.append(
        f"For relative group size, the estimated coefficient was {size_coef:.3f} "
        f"(odds ratio {size_or:.2f}, p-value {size_p:.3f}). "
        "A positive coefficient and odds ratio greater than 1 indicate that larger "
        "focal groups tend to have a higher probability of winning, whereas an odds "
        "ratio below 1 would suggest the opposite."
    )

    explanation_parts.append(
        f"For contest location advantage, the estimated coefficient was {loc_coef:.3f} "
        f"(odds ratio {loc_or:.2f}, p-value {loc_p:.3f}). "
        "A positive coefficient means that contests occurring closer to the focal group's "
        "home range center (relative to the other group) were associated with a higher "
        "probability of the focal group winning."
    )

    explanation_parts.append(
        "Using a conventional 0.05 significance threshold, I treated the variables as "
        "meaningfully influencing win probability when their p-values were below 0.05. "
        "I then mapped the overall strength of evidence that relative group size and/or "
        "contest location affect winning onto a 0–100 scale, where higher values "
        "correspond to stronger 'Yes' answers and lower values correspond to 'No' answers "
        "indicating little or no evidence of an effect."
    )

    if size_sig and loc_sig:
        explanation_parts.append(
            "Both relative group size and contest location showed statistically significant "
            "associations with winning, so I answered 'Yes' with a high confidence score "
            "on the scale."
        )
    elif size_sig or loc_sig:
        sig_var = "relative group size" if size_sig else "contest location"
        explanation_parts.append(
            f"{sig_var.capitalize()} showed a statistically significant association with winning, "
            "while the other predictor did not reach conventional significance in this sample. "
            "This provides moderate evidence that at least one of the two factors influences "
            "the probability of winning intergroup contests, leading to a 'Yes' answer with "
            "moderate confidence on the 0–100 scale."
        )
    else:
        explanation_parts.append(
            "Neither relative group size nor contest location showed statistically significant "
            "effects on win probability at the 0.05 level in this dataset, so I answered 'No' "
            "and assigned a relatively low score on the 0–100 scale to reflect the lack of "
            "strong evidence for an effect."
        )

    explanation_parts.append(
        "Because the dataset contains only 58 intergroup contests, the estimates are somewhat "
        "uncertain, and small-sample variability means that modest real effects could fail "
        "to reach statistical significance. The conclusion therefore emphasizes the evidence "
        "present in this dataset rather than ruling out any possible influence of group size "
        "or location in the broader population."
    )

    explanation = " ".join(explanation_parts)

    conclusion = {
        "response": int(response),
        "explanation": explanation,
    }

    conclusion_path = base_dir / "conclusion.txt"
    with conclusion_path.open("w", encoding="utf-8") as f:
        json.dump(conclusion, f, ensure_ascii=False)


if __name__ == "__main__":
    main()

