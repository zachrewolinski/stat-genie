import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main() -> None:
    data_path = Path("amtl.csv")
    df = pd.read_csv(data_path)

    # Rename for clarity
    df = df.rename(
        columns={
            "feature1": "tooth_class",
            "feature3": "missing",
            "feature4": "sockets",
            "feature5": "age",
            "feature7": "sex_estimate",
            "feature8": "genus",
        }
    )

    # Keep only rows with positive observable sockets (per metadata min is 2, but be safe)
    df = df[df["sockets"] > 0].copy()

    # Binary indicator for modern humans vs non-human primates
    df["is_human"] = (df["genus"] == "Homo sapiens").astype(int)

    # Overall AMTL proportions for descriptive context
    grouped = (
        df.groupby("is_human")[["missing", "sockets"]]
        .sum()
        .assign(prop_missing=lambda x: x["missing"] / x["sockets"])
    )

    human_prop = float(grouped.loc[1, "prop_missing"])
    nonhuman_prop = float(grouped.loc[0, "prop_missing"])

    # Fit binomial regression: AMTL (missing vs sockets) ~ human status + age + sex + tooth class
    df["prop_missing"] = df["missing"] / df["sockets"]

    model = smf.glm(
        formula="prop_missing ~ is_human + age + sex_estimate + C(tooth_class)",
        data=df,
        family=sm.families.Binomial(),
        freq_weights=df["sockets"],
    ).fit()

    coef = float(model.params["is_human"])
    p_value = float(model.pvalues["is_human"])
    odds_ratio = float(np.exp(coef))

    # Average marginal effect of human vs non-human status on AMTL probability
    base_df = df.copy()
    base_df["is_human"] = 0
    pred_nonhuman = model.predict(base_df).mean()
    base_df["is_human"] = 1
    pred_human = model.predict(base_df).mean()
    marginal_diff = float(pred_human - pred_nonhuman)

    # Decide response based on direction and significance of human effect
    alpha = 0.05
    if coef > 0 and p_value < alpha:
        response = "Yes"
    else:
        response = "No"

    explanation = (
        "I modeled the probability of antemortem tooth loss (AMTL) at the tooth level "
        "using a binomial regression of missing teeth out of observable sockets. "
        "The predictors were a binary indicator for modern humans (Homo sapiens vs. Pan/Papio/Pongo), "
        "estimated age at death, estimated sex, and tooth class (anterior, posterior, premolar). "
        f"Descriptively, humans had an overall AMTL proportion of {human_prop:.3f}, while non-human primates "
        f"had {nonhuman_prop:.3f}. In the regression model, the coefficient for the human indicator was "
        f"{coef:.3f}, corresponding to an odds ratio of {odds_ratio:.3f} (p = {p_value:.3f}). "
        f"The adjusted mean AMTL probability was {pred_human:.3f} for humans and {pred_nonhuman:.3f} for "
        f"non-human primates, a difference of {marginal_diff:.3f}. Based on the sign and statistical "
        "significance of the human effect after accounting for age, sex, and tooth class, "
        f"I therefore answer '{response}' to the research question."
    )

    conclusion = {"response": response, "explanation": explanation}

    with Path("conclusion.txt").open("w", encoding="utf-8") as f:
        json.dump(conclusion, f, ensure_ascii=False)


if __name__ == "__main__":
    main()

