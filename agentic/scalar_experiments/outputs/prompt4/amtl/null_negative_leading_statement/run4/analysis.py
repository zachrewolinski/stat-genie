import json
from pathlib import Path

import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main() -> None:
    # Load data
    df = pd.read_csv("amtl.csv")

    # Basic cleaning: drop rows with missing key fields and ensure positive socket counts
    df = df.dropna(
        subset=["num_amtl", "sockets", "age", "prob_male", "tooth_class", "genus"]
    )
    df = df[df["sockets"] > 0].copy()

    # Construct variables for analysis
    df["prop_missing"] = df["num_amtl"] / df["sockets"]
    df["is_human"] = (df["genus"] == "Homo sapiens").astype(int)

    # Descriptive summaries: overall AMTL proportions for humans vs non-humans
    human_mask = df["is_human"] == 1
    nonhuman_mask = df["is_human"] == 0

    human_amtl = df.loc[human_mask, "num_amtl"].sum()
    human_sockets = df.loc[human_mask, "sockets"].sum()
    nonhuman_amtl = df.loc[nonhuman_mask, "num_amtl"].sum()
    nonhuman_sockets = df.loc[nonhuman_mask, "sockets"].sum()

    human_prop = float(human_amtl) / float(human_sockets) if human_sockets > 0 else 0.0
    nonhuman_prop = (
        float(nonhuman_amtl) / float(nonhuman_sockets) if nonhuman_sockets > 0 else 0.0
    )

    # Fit a binomial regression model:
    #   logit(P(AMTL)) ~ is_human + age + prob_male + tooth_class
    # using sockets as binomial trial weights.
    df["age_centered"] = df["age"] - df["age"].mean()

    model = smf.glm(
        formula="prop_missing ~ is_human + age_centered + prob_male + C(tooth_class)",
        data=df,
        family=sm.families.Binomial(),
        freq_weights=df["sockets"],
    ).fit()

    coef_human = float(model.params.get("is_human", 0.0))
    pval_human = float(model.pvalues.get("is_human", 1.0))

    # Predicted probabilities for a representative tooth (Posterior) at mean age and sex
    mean_age_centered = 0.0
    mean_prob_male = float(df["prob_male"].mean())

    base_data = pd.DataFrame(
        {
            "is_human": [0, 1],
            "age_centered": [mean_age_centered, mean_age_centered],
            "prob_male": [mean_prob_male, mean_prob_male],
            "tooth_class": ["Posterior", "Posterior"],
        }
    )

    preds = model.predict(base_data)
    nonhuman_pred = float(preds.iloc[0])
    human_pred = float(preds.iloc[1])
    diff_pred = human_pred - nonhuman_pred

    # Map statistical evidence to a 0–100 Likert scale for the
    # question: "Do modern humans have higher AMTL frequencies
    # than non-human primates, after accounting for covariates?"
    # 0 = strong "No", 100 = strong "Yes".
    if coef_human > 0 and pval_human < 0.001:
        response = 95
    elif coef_human > 0 and pval_human < 0.01:
        response = 90
    elif coef_human > 0 and pval_human < 0.05:
        response = 80
    elif coef_human > 0 and pval_human < 0.1:
        response = 65
    elif coef_human > 0:
        response = 55
    elif coef_human < 0 and pval_human < 0.001:
        response = 5
    elif coef_human < 0 and pval_human < 0.01:
        response = 10
    elif coef_human < 0 and pval_human < 0.05:
        response = 20
    elif coef_human < 0 and pval_human < 0.1:
        response = 35
    elif coef_human < 0:
        response = 45
    else:
        response = 50

    # Build a human-readable explanation summarizing the evidence
    explanation_lines = []
    explanation_lines.append(
        "I fit a binomial regression model for the proportion of antemortem tooth loss "
        "(AMTL) at the tooth-class level, with logit(P(AMTL)) modeled as a function of "
        "a human-versus-nonhuman indicator, age, estimated sex, and tooth class using "
        "the counts of observable sockets as binomial trial weights."
    )
    explanation_lines.append(
        f"Unadjusted across all tooth classes, humans had {human_amtl} missing teeth "
        f"out of {human_sockets} observable sockets (proportion ≈ {human_prop:.3f}), "
        f"while non-human primates had {nonhuman_amtl} missing teeth out of "
        f"{nonhuman_sockets} sockets (proportion ≈ {nonhuman_prop:.3f})."
    )
    explanation_lines.append(
        "In the regression model that adjusts for age, estimated sex, and tooth class, "
        f"the coefficient for the human indicator was {coef_human:.3f} on the log-odds "
        f"scale with p-value {pval_human:.3g}."
    )
    explanation_lines.append(
        f"At representative values (mean age, mean estimated sex, posterior teeth), the "
        f"model predicts an AMTL probability of approximately {nonhuman_pred:.3f} for "
        f"non-human primates and {human_pred:.3f} for humans, a difference of "
        f"{diff_pred:.3f} in absolute probability."
    )
    if coef_human > 0:
        direction_sentence = (
            "These results indicate that, after accounting for age, sex, and tooth "
            "class, modern humans tend to have higher AMTL frequencies than the pooled "
            "non-human primate genera."
        )
    elif coef_human < 0:
        direction_sentence = (
            "These results indicate that, after accounting for age, sex, and tooth "
            "class, modern humans tend to have lower AMTL frequencies than the pooled "
            "non-human primate genera."
        )
    else:
        direction_sentence = (
            "The model does not show a clear directional difference in AMTL frequencies "
            "between humans and the pooled non-human primate genera after adjustment."
        )
    explanation_lines.append(direction_sentence)
    explanation_lines.append(
        f"Mapping this adjusted effect and its statistical strength onto a 0–100 Likert "
        f"scale for the question 'Do humans have higher AMTL than non-human primates?' "
        f"yields a response score of {response}, where values closer to 100 represent "
        f"stronger evidence that humans have higher AMTL and values closer to 0 "
        f"represent stronger evidence that they do not."
    )

    explanation = " ".join(explanation_lines)

    conclusion = {"response": int(response), "explanation": explanation}

    out_path = Path("conclusion.txt")
    out_path.write_text(json.dumps(conclusion), encoding="utf-8")


if __name__ == "__main__":
    main()

