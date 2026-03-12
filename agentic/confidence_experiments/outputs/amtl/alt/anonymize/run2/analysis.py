import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main() -> None:
    base_path = Path(__file__).parent
    data_path = base_path / "amtl.csv"
    info_path = base_path / "info.json"

    df = pd.read_csv(data_path)

    # Basic derived variables
    df["missing"] = df["feature3"]
    df["total"] = df["feature4"]
    df["amtl_rate"] = df["missing"] / df["total"]

    # Quick sanity-check summaries written to stdout for the analyst
    genus_summary = (
        df.groupby("feature8")
        .agg(
            n_rows=("feature2", "nunique"),
            total_sockets=("total", "sum"),
            total_missing=("missing", "sum"),
            mean_rate=("amtl_rate", "mean"),
        )
        .reset_index()
    )

    print("Genus-level AMTL summary:")
    print(genus_summary.to_string(index=False))
    print()

    # Build a socket-level dataset so that each tooth socket is one row
    # with a binary indicator of antemortem tooth loss (AMTL).
    records = []
    for _, row in df.iterrows():
        total = int(row["total"])
        missing = int(row["missing"])
        present = total - missing

        # One row per tooth socket: 1 = missing ante-mortem, 0 = present
        for outcome in ([1] * missing + [0] * present):
            records.append(
                {
                    "amtl": outcome,
                    "tooth_class": row["feature1"],
                    "age": row["feature5"],
                    "sex_est": row["feature7"],
                    "genus": row["feature8"],
                }
            )

    socket_df = pd.DataFrame.from_records(records)
    socket_df["is_human"] = (socket_df["genus"] == "Homo sapiens").astype(int)

    print("Socket-level dataset size:", len(socket_df))
    print(
        "Overall AMTL rate:",
        socket_df["amtl"].mean(),
    )
    print()

    # Fit a binomial (logistic) regression model for AMTL,
    # with a binary indicator for humans vs non-human primates,
    # adjusting for age, sex estimate, and tooth class.
    formula = "amtl ~ is_human + age + sex_est + C(tooth_class)"
    model = smf.glm(formula=formula, data=socket_df, family=sm.families.Binomial())
    result = model.fit()

    print(result.summary())
    print()

    # Extract the human vs non-human effect
    coef_human = result.params["is_human"]
    se_human = result.bse["is_human"]
    pval_human = result.pvalues["is_human"]
    ci_low, ci_high = result.conf_int().loc["is_human"]

    odds_ratio = float(np.exp(coef_human))
    or_ci_low = float(np.exp(ci_low))
    or_ci_high = float(np.exp(ci_high))

    # Predicted AMTL probabilities for a "typical" socket
    # (mean age, mean sex estimate, most common tooth class),
    # comparing humans vs non-humans.
    mean_age = float(socket_df["age"].mean())
    mean_sex = float(socket_df["sex_est"].mean())
    ref_tooth = socket_df["tooth_class"].mode()[0]

    pred_df = pd.DataFrame(
        {
            "is_human": [0, 1],
            "age": [mean_age, mean_age],
            "sex_est": [mean_sex, mean_sex],
            "tooth_class": [ref_tooth, ref_tooth],
        }
    )
    pred_probs = result.predict(pred_df)
    prob_nonhuman = float(pred_probs.iloc[0])
    prob_human = float(pred_probs.iloc[1])

    print("Predicted AMTL probability (non-human):", prob_nonhuman)
    print("Predicted AMTL probability (human):    ", prob_human)
    print("Odds ratio (human vs non-human):       ", odds_ratio)
    print("95% CI for odds ratio:                 ", (or_ci_low, or_ci_high))
    print("p-value for human effect:              ", pval_human)
    print()

    # Map the strength of evidence and effect size onto the 0-100 Likert scale.
    # Given the large observed difference in rates and (as seen above) the
    # extremely strong statistical evidence, we choose a value close to 100.
    if pval_human < 1e-8 and odds_ratio > 2.0:
        response_score = 98
    elif pval_human < 1e-4 and odds_ratio > 1.5:
        response_score = 90
    elif pval_human < 0.05 and odds_ratio > 1.2:
        response_score = 75
    else:
        response_score = 50

    # Build a concise textual explanation for conclusion.txt
    explanation_lines = [
        "Research question: Do modern humans (Homo sapiens) have higher "
        "frequencies of antemortem tooth loss (AMTL) than non-human primate "
        "genera (Pan, Pongo, Papio) after accounting for age, sex, and tooth class?",
        "",
        "Approach: I modeled AMTL at the level of individual tooth sockets "
        "using a binomial (logistic) regression. Each row in the original "
        "dataset provides the number of missing teeth and observable sockets "
        "for a given specimen and tooth class. I expanded these counts so that "
        "each socket became one binary observation (1 = antemortem tooth loss, "
        "0 = tooth present). The model included a binary indicator for humans "
        "vs non-humans, estimated age at death, sex estimate, and tooth class.",
        "",
        "Findings: After adjustment, the human indicator has a large positive "
        "coefficient (odds ratio {:.1f}, 95% CI {:.1f}–{:.1f}, p-value {:.1e}), "
        "indicating that human sockets are much more likely to show AMTL than "
        "non-human primate sockets of comparable age, sex estimate, and tooth class."
        .format(odds_ratio, or_ci_low, or_ci_high, pval_human),
        "For a typical socket (mean age and sex estimate, most common tooth "
        "class), the model predicts an AMTL probability of {:.3f} for "
        "non-human primates versus {:.3f} for humans."
        .format(prob_nonhuman, prob_human),
        "",
        "Conclusion: There is very strong statistical evidence that modern humans "
        "have substantially higher frequencies of antemortem tooth loss than "
        "non-human primates, even after accounting for age, sex, and tooth class. "
        "On a 0–100 scale, this corresponds to a strong 'Yes' answer with a "
        "response value of {}.".format(response_score),
    ]

    conclusion = {
        "response": int(response_score),
        "explanation": "\n".join(explanation_lines),
    }

    conclusion_path = base_path / "conclusion.txt"
    conclusion_path.write_text(json.dumps(conclusion))


if __name__ == "__main__":
    main()

