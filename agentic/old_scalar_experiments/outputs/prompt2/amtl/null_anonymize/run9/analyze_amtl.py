import json

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main() -> None:
    df = pd.read_csv("amtl.csv")

    df = df.copy()
    df["missing"] = df["feature3"].astype(float)
    df["total"] = df["feature4"].astype(float)

    df = df[(df["total"] > 0) & (df["missing"] >= 0) & (df["missing"] <= df["total"])].copy()
    df["missing_prop"] = df["missing"] / df["total"]

    df["is_human"] = (df["feature8"] == "Homo sapiens").astype(int)

    try:
        model = smf.glm(
            formula="missing_prop ~ is_human + feature5 + feature7 + C(feature1)",
            data=df,
            family=sm.families.Binomial(),
            freq_weights=df["total"],
        ).fit()

        coef = float(model.params["is_human"])
        pval = float(model.pvalues["is_human"])

        df["pred"] = model.predict(df)
        mean_pred_human = float(df.loc[df["is_human"] == 1, "pred"].mean())
        mean_pred_nonhuman = float(df.loc[df["is_human"] == 0, "pred"].mean())
        diff = mean_pred_human - mean_pred_nonhuman

        if diff > 0 and pval < 0.05:
            response = "Yes"
        else:
            response = "No"

        logp = -np.log10(pval + 1e-12)
        effect_scale = min(max(abs(diff) * 200.0, 0.0), 20.0)
        conf_score = 50.0 + min(logp * 5.0, 40.0) + effect_scale
        confidence = int(round(max(0.0, min(100.0, conf_score))))

        explanation_parts = [
            "Fitted a binomial regression (logit link) modeling the proportion of missing teeth ",
            "(feature3 / feature4) as a function of a human-vs-nonhuman indicator, age at death ",
            "(feature5), estimated sex (feature7), and tooth class (feature1), using each specimen's ",
            "number of observable sockets as frequency weights. ",
            f"The estimated coefficient for humans (is_human=1) was {coef:.3f} with p-value {pval:.3g}. ",
            "Model-based average predicted antemortem tooth loss (AMTL) frequency was ",
            f"{mean_pred_human:.3f} for modern humans and {mean_pred_nonhuman:.3f} for non-human primates, ",
            f"yielding a difference of {diff:.3f}. ",
        ]

        if response == "Yes":
            explanation_parts.append(
                "Because the human coefficient is positive and statistically significant and the "
                "covariate-adjusted predicted AMTL frequency is higher in humans than in non-human genera, "
                "this analysis supports the conclusion that modern humans have higher AMTL frequencies after "
                "accounting for age, sex, and tooth class."
            )
        else:
            explanation_parts.append(
                "Because the human coefficient is not clearly positive and statistically significant or the "
                "covariate-adjusted predicted AMTL frequency is not higher in humans, this analysis does not "
                "support the conclusion that modern humans have higher AMTL frequencies after accounting for "
                "age, sex, and tooth class."
            )

        explanation = "".join(explanation_parts)

    except Exception as e:
        human = df[df["is_human"] == 1]
        nonhuman = df[df["is_human"] == 0]

        rate_human = float(human["missing"].sum() / human["total"].sum())
        rate_nonhuman = float(nonhuman["missing"].sum() / nonhuman["total"].sum())
        diff = rate_human - rate_nonhuman

        response = "Yes" if diff > 0 else "No"
        confidence = 65

        explanation = (
            "Statistical model fitting failed ("
            f"{e}"
            "). As a fallback, compared the overall proportion of missing teeth between humans and non-human "
            "primates, pooling across age, sex, and tooth class. "
            f"The overall AMTL proportion was {rate_human:.3f} for modern humans and {rate_nonhuman:.3f} for "
            f"non-human primates, for a difference of {diff:.3f}. "
        )

        if response == "Yes":
            explanation += (
                "Because humans show a higher pooled AMTL proportion, this suggests—but does not strongly "
                "confirm—that humans have higher AMTL frequencies."
            )
        else:
            explanation += (
                "Because humans do not show a higher pooled AMTL proportion, this suggests that humans do not "
                "have higher AMTL frequencies."
            )

    conclusion = {
        "response": response,
        "confidence": confidence,
        "explanation": explanation,
    }

    with open("conclusion.txt", "w") as f:
        json.dump(conclusion, f)


if __name__ == "__main__":
    main()

