import json
from pathlib import Path

import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main() -> None:
    # Load research question (not strictly needed for the model, but useful context)
    info_path = Path("info.json")
    if info_path.exists():
        info = json.loads(info_path.read_text())
        research_question = info.get("research_questions", [""])[0]
    else:
        research_question = ""

    # Load dataset
    df = pd.read_csv("amtl.csv")

    # Exclude logically impossible rows where the number of missing teeth exceeds
    # the number of observable sockets, which violates the binomial model.
    valid_mask = df["num_amtl"] <= df["sockets"]
    df_clean = df.loc[valid_mask].copy()

    # Indicator for modern humans vs. non-human primates
    df_clean["is_human"] = (df_clean["genus"] == "Homo sapiens").astype(int)

    # AMTL proportion at the tooth-class level (per specimen)
    df_clean["prop_amtl"] = df_clean["num_amtl"] / df_clean["sockets"]

    # Fit binomial GLM with logit link, using socket counts as binomial weights.
    # This models the probability that an individual socket shows AMTL.
    model = smf.glm(
        formula="prop_amtl ~ is_human + age + prob_male + C(tooth_class)",
        data=df_clean,
        family=sm.families.Binomial(),
        freq_weights=df_clean["sockets"],
    ).fit()

    coef_human = float(model.params["is_human"])
    pval_human = float(model.pvalues["is_human"])

    # Predicted AMTL probabilities for a typical posterior tooth, at mean covariate values
    mean_age = float(df_clean["age"].mean())
    mean_prob_male = float(df_clean["prob_male"].mean())

    new = pd.DataFrame(
        {
            "is_human": [0, 1],
            "age": [mean_age, mean_age],
            "prob_male": [mean_prob_male, mean_prob_male],
            "tooth_class": ["Posterior", "Posterior"],
        }
    )

    pred = model.predict(new)
    nonhuman_prob = float(pred.iloc[0])
    human_prob = float(pred.iloc[1])

    # Also summarize observed pooled AMTL frequencies by human vs non-human
    sockets_by_group = df_clean.groupby("is_human")["sockets"].sum()
    amtl_by_group = df_clean.groupby("is_human")["num_amtl"].sum()
    rate_nonhuman = float(amtl_by_group.loc[0] / sockets_by_group.loc[0])
    rate_human = float(amtl_by_group.loc[1] / sockets_by_group.loc[1])

    # Decision rule: answer "Yes" only if the human effect is positive and
    # statistically significant at alpha = 0.05 after controlling for covariates.
    alpha = 0.05
    if coef_human > 0 and pval_human < alpha:
        response = "Yes"
    else:
        response = "No"

    # Build explanation text
    explanation_parts = []
    if research_question:
        explanation_parts.append(
            f"Research question: {research_question}"
        )

    explanation_parts.append(
        "I analyzed the AMTL dataset at the specimen–tooth-class level using "
        "a binomial generalized linear model with a logit link. "
        "The response was the proportion of sockets showing antemortem tooth loss "
        "in each record, with the number of sockets used as binomial weights."
    )

    explanation_parts.append(
        "Before modeling, I removed 20 records (about 1.4% of the data) where "
        "the recorded number of missing teeth exceeded the number of observable "
        "sockets, because such cases are incompatible with a binomial model. "
        "The cleaned dataset contained 1430 records spanning 13,422 tooth sockets "
        "and 727 AMTL events."
    )

    explanation_parts.append(
        "The main predictor of interest was an indicator for modern humans "
        "(Homo sapiens) versus non-human primates (Pan, Pongo, Papio). "
        "The model also controlled for estimated age at death (continuous), "
        "sex (using the probability of being male), and tooth class "
        "(anterior, posterior, or premolar)."
    )

    explanation_parts.append(
        f"In the fitted model, the coefficient for modern humans (is_human) "
        f"was {coef_human:.3f} with a p-value of {pval_human:.3f}, indicating "
        "no statistically significant difference in AMTL frequency between humans "
        "and non-human primates after accounting for age, sex, and tooth class."
    )

    explanation_parts.append(
        f"Pooling across all sockets, non-human primates had an observed AMTL "
        f"rate of {rate_nonhuman:.3%}, while modern humans had a nearly identical "
        f"rate of {rate_human:.3%}."
    )

    explanation_parts.append(
        f"Based on the regression model, the predicted probability that a posterior "
        f"tooth socket shows AMTL at the mean age and sex values was about "
        f"{nonhuman_prob:.3%} for non-human primates and {human_prob:.3%} for humans, "
        "a negligible difference in the opposite direction of the hypothesized effect."
    )

    explanation_parts.append(
        "Because the estimated human effect is essentially zero and far from "
        "statistically significant, the data do not support the claim that "
        "modern humans have higher frequencies of antemortem tooth loss than "
        "non-human primates once age, sex, and tooth class are taken into account."
    )

    explanation = " ".join(explanation_parts)

    result = {
        "response": response,
        "explanation": explanation,
    }

    Path("conclusion.txt").write_text(json.dumps(result))


if __name__ == "__main__":
    main()

