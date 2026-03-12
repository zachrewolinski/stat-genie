import json
import math

import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main() -> None:
    # Load dataset
    df = pd.read_csv("amtl.csv")

    # Keep only variables needed for the analysis
    cols = ["tooth_class", "num_amtl", "sockets", "age", "prob_male", "genus"]
    df = df[cols].copy()

    # Basic cleaning: drop missing values and any rows with non-positive socket counts
    df = df.dropna(subset=["tooth_class", "num_amtl", "sockets", "age", "prob_male", "genus"])
    df = df[df["sockets"] > 0]

    # AMTL rate per row and indicator for modern humans
    df["amtl_rate"] = df["num_amtl"] / df["sockets"]
    df["is_human"] = (df["genus"].astype(str).str.strip() == "Homo sapiens").astype(int)

    # Weighted mean AMTL rates (weight by number of sockets)
    grouped = df.groupby("is_human").apply(
        lambda g: g["num_amtl"].sum() / g["sockets"].sum()
    )
    human_rate = float(grouped.get(1, float("nan")))
    nonhuman_rate = float(grouped.get(0, float("nan")))

    # Binomial regression: AMTL proportion with binomial family and socket counts as weights
    model = smf.glm(
        formula="amtl_rate ~ is_human + age + prob_male + C(tooth_class)",
        data=df,
        family=sm.families.Binomial(),
        freq_weights=df["sockets"],
    )
    result = model.fit()

    human_coef = float(result.params["is_human"])
    human_se = float(result.bse["is_human"])
    human_p = float(result.pvalues["is_human"])

    ci_low, ci_high = result.conf_int().loc["is_human"]
    ci_low = float(ci_low)
    ci_high = float(ci_high)

    or_human = float(math.exp(human_coef))
    or_ci_low = float(math.exp(ci_low))
    or_ci_high = float(math.exp(ci_high))

    # Map evidence to a 0–100 Likert-style response
    if human_p > 0.05:
        qualitative = "No"
        response = 30
    else:
        qualitative = "Yes"
        if human_p < 0.001:
            base = 90
        elif human_p < 0.01:
            base = 80
        else:  # 0.01 <= p <= 0.05
            base = 70

        # Adjust confidence based on effect size (odds ratio)
        # Large positive effects push the response up, small or near-null effects closer to base.
        adjust = max(-10.0, min(10.0, (or_human - 1.0) * 10.0))
        response = int(round(base + adjust))

    response = max(0, min(100, int(response)))

    explanation = (
        "Research question: Do modern humans (Homo sapiens) have higher frequencies of "
        "antemortem tooth loss (AMTL) than non-human primates (Pan, Pongo, Papio) after "
        "accounting for age, sex, and tooth class? "
        "I modeled the number of missing teeth out of observable sockets using a binomial "
        "regression with predictors for a human-versus-non-human indicator, age at death, "
        "probability of being male, and tooth class (anterior, posterior, premolar). "
        f"Across the dataset, humans had an overall AMTL rate of {human_rate:.3f} missing teeth "
        f"per socket, while non-human primates had a rate of {nonhuman_rate:.3f}. "
        f"In the regression model, the human indicator had an estimated log-odds coefficient of "
        f"{human_coef:.3f} (SE = {human_se:.3f}, p = {human_p:.3g}), corresponding to an odds "
        f"ratio of {or_human:.2f} with a 95% confidence interval from {or_ci_low:.2f} to "
        f"{or_ci_high:.2f}. "
    )

    if qualitative == "Yes":
        explanation += (
            "Because the human indicator is positive and statistically significant after "
            "controlling for age, sex, and tooth class, the analysis supports the conclusion "
            "that modern humans have higher AMTL frequencies than the non-human primate genera "
            "in this sample. "
        )
    else:
        explanation += (
            "Because the human indicator is not statistically significant after controlling for "
            "age, sex, and tooth class, there is insufficient evidence that modern humans differ "
            "in AMTL frequency from the non-human primate genera in this sample. "
        )

    explanation += (
        f"I therefore answer \"{qualitative}\" to the research question and map this to a "
        f"response value of {response} on a 0–100 scale, where higher values indicate stronger "
        "evidence in favor of humans having higher AMTL rates."
    )

    output = {"response": int(response), "explanation": explanation}

    with open("conclusion.txt", "w") as f:
        json.dump(output, f)

    # Print a brief summary for interactive inspection (does not affect conclusion.txt)
    print("Human vs non-human weighted AMTL rates:", human_rate, nonhuman_rate)
    print("is_human coef, SE, p:", human_coef, human_se, human_p)
    print("OR (95% CI):", or_human, (or_ci_low, or_ci_high))
    print("Likert response:", response)


if __name__ == "__main__":
    main()
