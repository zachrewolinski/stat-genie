import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main() -> None:
    data_path = Path("amtl.csv")
    df = pd.read_csv(data_path)

    # Keep only rows with valid socket counts
    df = df[df["sockets"] > 0].copy()

    # Indicator for modern humans vs non-human primates
    df["is_human"] = (df["genus"] == "Homo sapiens").astype(int)

    # Proportion of missing teeth in the tooth class
    df["amtl_prop"] = df["num_amtl"] / df["sockets"]

    # Binomial regression: probability a tooth is missing in a class
    # num_amtl successes out of sockets trials
    formula = "amtl_prop ~ is_human + age + prob_male + C(tooth_class)"
    model = smf.glm(
        formula=formula,
        data=df,
        family=sm.families.Binomial(),
        freq_weights=df["sockets"],
    )
    result = model.fit()

    # Extract human effect
    coef_human = float(result.params["is_human"])
    se_human = float(result.bse["is_human"])
    p_human = float(result.pvalues["is_human"])
    ci_low, ci_high = result.conf_int().loc["is_human"]

    odds_ratio = float(np.exp(coef_human))
    or_ci_low = float(np.exp(ci_low))
    or_ci_high = float(np.exp(ci_high))

    # Predicted probabilities for a typical specimen across tooth classes
    mean_age = float(df["age"].mean())
    mean_prob_male = float(df["prob_male"].mean())
    tooth_classes = sorted(df["tooth_class"].unique())

    scenarios = []
    for tc in tooth_classes:
        for is_human in (0, 1):
            scenarios.append(
                {
                    "is_human": is_human,
                    "age": mean_age,
                    "prob_male": mean_prob_male,
                    "tooth_class": tc,
                }
            )
    scen_df = pd.DataFrame(scenarios)
    scen_df["pred_prob"] = result.predict(scen_df)

    mean_pred_nonhuman = float(
        scen_df[scen_df["is_human"] == 0]["pred_prob"].mean()
    )
    mean_pred_human = float(
        scen_df[scen_df["is_human"] == 1]["pred_prob"].mean()
    )

    # Observed genus-level proportions of missing teeth
    obs = (
        df.groupby("genus")
        .agg(total_missing=("num_amtl", "sum"), total_sockets=("sockets", "sum"))
        .assign(prop_missing=lambda d: d["total_missing"] / d["total_sockets"])
    )
    genus_summaries = []
    for genus, row in obs.iterrows():
        genus_summaries.append(
            f"{genus}: {row['prop_missing']:.3%} "
            f"(missing {int(row['total_missing'])} of {int(row['total_sockets'])} sockets)"
        )
    genus_summary_str = "; ".join(genus_summaries)

    # Decide on the answer: focus on sign and significance of human effect
    human_effect_positive = coef_human > 0
    human_effect_significant = p_human < 0.05

    if human_effect_positive and human_effect_significant:
        response = "Yes"
        interpretation = (
            "After adjusting for age, sex (probability of being male), and tooth class, "
            "modern humans have higher odds of antemortem tooth loss than the non-human primates."
        )
    else:
        response = "No"
        interpretation = (
            "After adjusting for age, sex (probability of being male), and tooth class, "
            "the model does not provide strong evidence that modern humans have higher odds "
            "of antemortem tooth loss than the non-human primates."
        )

    explanation = (
        f"{interpretation} "
        f"In a binomial regression of the number of missing teeth out of the number of "
        f"observable sockets, the coefficient for the indicator of being a modern human "
        f"(vs. non-human primate) was {coef_human:.3f} on the log-odds scale "
        f"(SE = {se_human:.3f}, p = {p_human:.4f}), corresponding to an odds ratio of "
        f"{odds_ratio:.2f} with a 95% confidence interval of {or_ci_low:.2f} to {or_ci_high:.2f}. "
        f"For a typical specimen (average age and sex estimate) averaged across tooth classes, "
        f"the model-predicted probability that a tooth is missing was "
        f"{mean_pred_human:.3%} for humans and {mean_pred_nonhuman:.3%} for non-human primates. "
        f"Observed genus-level missing-tooth proportions were: {genus_summary_str}. "
        f"These results summarize how AMTL frequency differs between modern humans and the three "
        f"non-human primate genera (Pan, Pongo, Papio) while accounting for the specified covariates."
    )

    conclusion = {"response": response, "explanation": explanation}

    with Path("conclusion.txt").open("w", encoding="utf-8") as f:
        json.dump(conclusion, f, ensure_ascii=False)


if __name__ == "__main__":
    main()

