import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def main() -> None:
    data_path = Path("affairs.csv")
    df = pd.read_csv(data_path)

    # Binary outcome: any extramarital intercourse in past year
    df["has_affair"] = (df["feature2"] > 0).astype(int)

    # Descriptive statistics by children status (feature6: "yes"/"no")
    group = df.groupby("feature6", observed=True)
    counts = group["has_affair"].size()
    affair_rate = group["has_affair"].mean()
    freq_mean = group["feature2"].mean()

    # Logistic regression controlling for observed covariates
    # C(feature6) captures effect of having children; reference is the first
    # level (alphabetical), which is expected to be "no".
    formula = (
        "has_affair ~ C(feature6) + C(feature3) + feature4 + "
        "feature5 + feature7 + feature8 + feature9 + feature10"
    )
    logit_model = smf.logit(formula, data=df)
    result = logit_model.fit(disp=False)

    # Extract children effect coefficient and p-value
    child_param = None
    for name in result.params.index:
        if "C(feature6)" in name:
            child_param = name
            break

    if child_param is None:
        raise RuntimeError("Could not locate children parameter in model.")

    child_coef = float(result.params[child_param])
    child_p = float(result.pvalues[child_param])

    # Decide answer:
    # Treat as evidence of a decrease only if the effect is negative
    # and statistically significant at the 5% level.
    if child_coef < 0 and child_p < 0.05:
        response = "Yes"
    else:
        response = "No"

    # Build explanation string with key numerical evidence
    # Ensure stable ordering for children/no-children groups
    groups = sorted(counts.index.tolist())

    parts = []
    parts.append(
        "I modeled the probability of engaging in any extramarital intercourse "
        "in the past year as a logistic regression on the survey data "
        "(601 married respondents), with a binary outcome indicating whether "
        "feature2 (self-reported frequency of extramarital intercourse) was "
        "greater than zero."
    )

    desc_segments = []
    for g in groups:
        n = int(counts[g])
        rate = float(affair_rate[g])
        mean_freq = float(freq_mean[g])
        desc_segments.append(
            f"{g} (n={n}): share with any affair ≈ {rate:.3f}, "
            f"mean affair-frequency code ≈ {mean_freq:.3f}"
        )
    parts.append(
        "By children status (feature6 = 'yes'/'no'), the descriptive "
        "statistics were: " + "; ".join(desc_segments) + "."
    )

    parts.append(
        "The logistic regression included children status (feature6), gender "
        "(feature3), age (feature4), years married (feature5), religiousness "
        "(feature7), education (feature8), occupation (feature9), and "
        "self-rated marriage quality (feature10) as predictors."
    )

    parts.append(
        f"In this model, the coefficient on having children "
        f"({child_param}) was {child_coef:.3f} with p-value {child_p:.3f}."
    )

    if response == "Yes":
        parts.append(
            "Because this coefficient is negative and statistically significant "
            "at the 5% level, the data provide evidence that having children "
            "is associated with a lower probability of engaging in extramarital "
            "affairs, after adjusting for the other recorded covariates."
        )
    else:
        direction = "negative" if child_coef < 0 else "positive"
        parts.append(
            "Although the estimated coefficient is "
            f"{direction}, its p-value exceeds the conventional 5% threshold, "
            "so the association between having children and the probability of "
            "extramarital affairs is not statistically distinguishable from zero "
            "once we control for the other covariates in the model."
        )
        parts.append(
            "Taken together, the descriptive differences and the regression "
            "results do not provide strong evidence that having children "
            "reduces engagement in extramarital affairs in this sample."
        )

    explanation = " ".join(parts)

    conclusion = {"response": response, "explanation": explanation}
    with open("conclusion.txt", "w", encoding="utf-8") as f:
        json.dump(conclusion, f, ensure_ascii=False)


if __name__ == "__main__":
    main()

