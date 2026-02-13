import json
from pathlib import Path

import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main() -> None:
    # Load dataset
    data_path = Path("amtl.csv")
    df = pd.read_csv(data_path)

    # Proportion of missing teeth in each specimen/tooth-class segment
    df = df[df["sockets"] > 0].copy()
    df["prop_missing"] = df["num_amtl"] / df["sockets"]

    # Indicator for modern humans vs non-human primates
    df["is_human"] = (df["genus"] == "Homo sapiens").astype(int)

    # Binomial regression: AMTL proportion ~ human vs non-human + age + sex + tooth class
    model = smf.glm(
        formula="prop_missing ~ is_human + age + prob_male + C(tooth_class)",
        data=df,
        family=sm.families.Binomial(),
        freq_weights=df["sockets"],
    ).fit()

    # Key effect: humans vs non-humans
    coef = float(model.params["is_human"])
    pval = float(model.pvalues["is_human"])
    lower_ci, upper_ci = (float(x) for x in model.conf_int().loc["is_human"])

    # Descriptive AMTL rates by genus
    totals = df.groupby("genus").agg(
        total_amtl=("num_amtl", "sum"),
        total_sockets=("sockets", "sum"),
    )
    totals["rate"] = totals["total_amtl"] / totals["total_sockets"]

    human_label = "Homo sapiens"
    human_rate = float(totals.loc[human_label, "rate"]) if human_label in totals.index else float("nan")

    nonhuman_genera = [g for g in ["Pan", "Papio", "Pongo"] if g in totals.index]
    if nonhuman_genera:
        nonhuman_amtl = totals.loc[nonhuman_genera, "total_amtl"].sum()
        nonhuman_sockets = totals.loc[nonhuman_genera, "total_sockets"].sum()
        nonhuman_rate = float(nonhuman_amtl / nonhuman_sockets) if nonhuman_sockets > 0 else float("nan")
    else:
        nonhuman_rate = float("nan")

    # Decision rule: "Yes" only if humans have a significantly *higher* AMTL rate
    alpha = 0.05
    is_higher = coef > 0 and (pval / 2) < alpha
    response = "Yes" if is_higher else "No"

    # Build explanation text
    explanation_parts = []

    explanation_parts.append(
        "I analyzed the antemortem tooth loss (AMTL) dataset using a binomial "
        "logistic regression. For each specimen and tooth-class combination, I "
        "modeled the proportion of missing teeth (num_amtl / sockets) with a "
        "logit link, using the number of observable sockets as frequency weights."
    )

    explanation_parts.append(
        "The main predictor of interest was a binary indicator for modern humans "
        "(is_human = 1 for Homo sapiens, 0 for non-human primates: Pan, Papio, "
        "and Pongo). The model also included covariates for estimated age at "
        "death, probability of being male (prob_male), and categorical tooth "
        "class (anterior, posterior, premolar), so that genus differences are "
        "evaluated after accounting for age, sex, and tooth class."
    )

    explanation_parts.append(
        f"In this regression, the estimated coefficient for humans (is_human) on "
        f"the log-odds scale was {coef:.3f} with a 95% confidence interval of "
        f"[{lower_ci:.3f}, {upper_ci:.3f}] and a two-sided p-value of "
        f"{pval:.4f}."
    )

    if is_higher:
        explanation_parts.append(
            "This coefficient is positive and statistically significant in a "
            "one-sided test for higher human AMTL (p_one_sided < 0.05), "
            "indicating that, after adjusting for age, sex, and tooth class, "
            "modern humans have higher odds of AMTL than the pooled non-human "
            "primates."
        )
    else:
        explanation_parts.append(
            "This coefficient is not significantly positive in a one-sided test "
            "for higher human AMTL: the 95% confidence interval includes zero "
            "and/or the point estimate is negative. Thus, the data do not "
            "support the claim that modern humans have higher AMTL frequencies "
            "than non-human primates once age, sex, and tooth class are taken "
            "into account."
        )

    if not pd.isna(human_rate) and not pd.isna(nonhuman_rate):
        explanation_parts.append(
            "To complement the regression, I compared overall weighted AMTL "
            "rates (total missing teeth divided by total observable sockets). "
            f"The pooled rate for Homo sapiens was {human_rate:.3f}, while the "
            f"pooled rate for the non-human genera (Pan, Papio, and Pongo) was "
            f"{nonhuman_rate:.3f}."
        )

        if human_rate > nonhuman_rate and not is_higher:
            explanation_parts.append(
                "Although the raw human rate is numerically higher, the "
                "regression analysis adjusting for age, sex, and tooth class "
                "does not show a statistically robust human excess in AMTL."
            )
        elif human_rate <= nonhuman_rate:
            explanation_parts.append(
                "Even at the descriptive level, humans do not show higher AMTL "
                "frequencies than the pooled non-human primates."
            )

    explanation_parts.append(
        f"Based on this analysis, my answer to the question "
        f"\"Do modern humans (Homo sapiens) have higher frequencies of "
        f"antemortem tooth loss than non-human primates (Pan, Pongo, Papio) "
        f"after accounting for age, sex, and tooth class?\" is \"{response}\"."
    )

    explanation = " ".join(explanation_parts)

    result = {"response": response, "explanation": explanation}

    with open("conclusion.txt", "w") as f:
        json.dump(result, f)


if __name__ == "__main__":
    main()

