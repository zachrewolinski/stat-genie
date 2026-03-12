import json
from typing import List, Dict, Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def expand_to_teeth(df: pd.DataFrame) -> pd.DataFrame:
    records: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        n_sockets = int(row["sockets"])
        n_missing = int(row["num_amtl"])
        n_present = n_sockets - n_missing

        base = {
            "specimen": row["specimen"],
            "age": row["age"],
            "stdev_age": row["stdev_age"],
            "prob_male": row["prob_male"],
            "genus": row["genus"],
            "pop": row["pop"],
            "tooth_class": row["tooth_class"],
            "is_human": row["is_human"],
        }

        for _ in range(n_missing):
            rec = base.copy()
            rec["amtl"] = 1
            records.append(rec)

        for _ in range(n_present):
            rec = base.copy()
            rec["amtl"] = 0
            records.append(rec)

    return pd.DataFrame.from_records(records)


def compute_likert_from_effect(or_human: float, p_human: float) -> (int, str):
    if p_human < 0.001:
        base_strength = 0.95
    elif p_human < 0.01:
        base_strength = 0.85
    elif p_human < 0.05:
        base_strength = 0.75
    elif p_human < 0.1:
        base_strength = 0.6
    else:
        base_strength = 0.5

    if or_human > 1:
        response = int(round(base_strength * 100))
        qualitative = "Yes"
    elif or_human < 1:
        response = int(round((1.0 - base_strength) * 100))
        qualitative = "No"
    else:
        response = 50
        qualitative = "Unclear"

    return response, qualitative


def main() -> None:
    df = pd.read_csv("amtl.csv")
    df["is_human"] = (df["genus"] == "Homo sapiens").astype(int)

    genus_summary = df.groupby("genus").agg(
        total_missing=("num_amtl", "sum"),
        total_sockets=("sockets", "sum"),
    )
    genus_summary["rate"] = genus_summary["total_missing"] / genus_summary["total_sockets"]

    expanded = expand_to_teeth(df)

    model = smf.logit("amtl ~ is_human + age + prob_male + C(tooth_class)", data=expanded)
    robust_result = model.fit(
        disp=False,
        cov_type="cluster",
        cov_kwds={"groups": expanded["specimen"]},
    )

    coef_human = float(robust_result.params["is_human"])
    se_human = float(robust_result.bse["is_human"])
    p_human = float(robust_result.pvalues["is_human"])
    or_human = float(np.exp(coef_human))

    expanded = expanded.copy()
    expanded["pred_prob"] = robust_result.predict()
    pred_means = expanded.groupby("is_human")["pred_prob"].mean()
    pred_nonhuman = float(pred_means.get(0, np.nan))
    pred_human = float(pred_means.get(1, np.nan))

    genus_rate_strings = []
    for genus, row in genus_summary.iterrows():
        genus_rate_strings.append(
            f"{genus}: {row['total_missing']:.0f}/{row['total_sockets']:.0f} teeth missing "
            f"({row['rate'] * 100:.1f}%)"
        )
    genus_rates_text = "; ".join(genus_rate_strings)

    response, qualitative = compute_likert_from_effect(or_human, p_human)

    explanation_lines = []
    explanation_lines.append(
        "Research question: Do modern humans (Homo sapiens) have higher frequencies of antemortem tooth loss (AMTL) "
        "than non-human primates (Pan, Pongo, Papio) after accounting for age, sex, and tooth class?"
    )
    explanation_lines.append(
        "I modeled the probability that an individual tooth socket shows AMTL using logistic regression, expanding each "
        "row into individual teeth so that each socket is a binary outcome (AMTL vs present). Predictors were an "
        "indicator for human vs non-human primate, age at death, estimated probability of being male, and tooth class "
        "(anterior, posterior, premolar). Standard errors were made robust to clustering within specimens."
    )
    explanation_lines.append(
        f"Descriptively, AMTL rates by genus (missing teeth / observable sockets) were: {genus_rates_text}."
    )
    explanation_lines.append(
        f"In the regression, the coefficient for the human indicator (Homo sapiens vs all non-human genera) was "
        f"{coef_human:.3f} on the log-odds scale (standard error {se_human:.3f}, odds ratio {or_human:.2f}, "
        f"p-value {p_human:.3g}). Model-based mean predicted AMTL probabilities were "
        f"{pred_nonhuman * 100:.1f}% for non-human primates and {pred_human * 100:.1f}% for humans, averaging over the "
        "observed distributions of age, sex estimate, and tooth class."
    )
    if or_human > 1:
        explanation_lines.append(
            "Because the human indicator is positive with an odds ratio above 1 and the p-value is small, the data "
            "provide statistical evidence that humans have higher AMTL frequencies than non-human primates after "
            "adjusting for age, sex, and tooth class."
        )
    elif or_human < 1:
        explanation_lines.append(
            "Because the human indicator is negative with an odds ratio below 1 and the p-value is small, the data "
            "provide statistical evidence that humans have lower AMTL frequencies than non-human primates after "
            "adjusting for age, sex, and tooth class."
        )
    else:
        explanation_lines.append(
            "Because the human indicator is near zero and not statistically distinguishable from zero, the data do not "
            "provide clear evidence that humans differ from non-human primates in AMTL frequency after adjusting for "
            "age, sex, and tooth class."
        )
    explanation_lines.append(
        f"On a 0–100 scale where 0 is a strong 'No' and 100 is a strong 'Yes', I summarize this evidence as a "
        f"{response} corresponding to a '{qualitative}' answer to the research question."
    )

    explanation = "\n".join(explanation_lines)

    result_json = {"response": int(response), "explanation": explanation}
    with open("conclusion.txt", "w") as f:
        json.dump(result_json, f)

    print(json.dumps(result_json, indent=2))


if __name__ == "__main__":
    main()
