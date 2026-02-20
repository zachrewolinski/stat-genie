import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def main() -> None:
    data_path = Path("amtl.csv")
    df = pd.read_csv(data_path)

    # Basic cleaning: drop rows with missing key fields and enforce integer counts.
    df = df.dropna(subset=["num_amtl", "sockets", "age", "prob_male", "genus", "tooth_class"])
    df = df[df["sockets"] > 0]
    df["sockets"] = df["sockets"].round().astype(int)
    df["num_amtl"] = df["num_amtl"].round().astype(int)

    # Expand to one row per tooth socket so we can model a binary AMTL outcome.
    rows = []
    for _, r in df.iterrows():
        sockets = int(r["sockets"])
        num_amtl = int(r["num_amtl"])
        if sockets <= 0:
            continue
        if num_amtl < 0:
            num_amtl = 0
        if num_amtl > sockets:
            num_amtl = sockets

        for i in range(sockets):
            rows.append(
                {
                    "amtl": 1 if i < num_amtl else 0,
                    "age": r["age"],
                    "prob_male": r["prob_male"],
                    "genus": r["genus"],
                    "tooth_class": r["tooth_class"],
                }
            )

    long_df = pd.DataFrame(rows)

    # Ensure categorical coding with Homo sapiens as the reference genus.
    long_df["genus"] = long_df["genus"].astype("category")
    genus_cats = list(long_df["genus"].cat.categories)
    if "Homo sapiens" in genus_cats:
        genus_cats = ["Homo sapiens"] + [g for g in genus_cats if g != "Homo sapiens"]
    long_df["genus"] = long_df["genus"].cat.reorder_categories(genus_cats, ordered=False)

    long_df["tooth_class"] = long_df["tooth_class"].astype("category")

    # Fit logistic regression for AMTL with genus, age, sex, and tooth class as predictors.
    model = smf.logit("amtl ~ C(genus) + age + prob_male + C(tooth_class)", data=long_df).fit(disp=False)

    params = model.params
    conf_int = model.conf_int()

    genus_terms = [name for name in params.index if name.startswith("C(genus)[")]
    all_others_lower = True
    genus_details = []

    for term in genus_terms:
        lower, upper = conf_int.loc[term]
        coef = params[term]
        odds_ratio = float(np.exp(coef))
        genus_name = term.split("[", 1)[1].split("]", 1)[0]
        genus_details.append(
            {
                "genus": genus_name,
                "coef": float(coef),
                "lower": float(lower),
                "upper": float(upper),
                "odds_ratio": odds_ratio,
            }
        )
        # Coefficient is log-odds difference for this genus vs Homo sapiens.
        # If the upper CI is >= 0, we cannot say this genus has definitively lower AMTL than humans.
        if upper >= 0:
            all_others_lower = False

    response = "Yes" if all_others_lower else "No"

    # Summarize unadjusted genus-level AMTL frequencies for context.
    genus_summary = (
        long_df.groupby("genus")["amtl"]
        .agg(["mean", "sum", "count"])
        .reset_index()
        .rename(columns={"mean": "prop_amtl", "sum": "num_amtl_sockets", "count": "total_sockets"})
    )

    lines = []
    lines.append(
        "I fit a logistic regression model predicting the probability that an individual tooth socket shows "
        "antemortem tooth loss (AMTL) from genus, age at death, sex (encoded as the probability of being male), "
        "and tooth class (anterior, posterior, premolar), using one row per observable socket."
    )
    lines.append(
        "In this model, Homo sapiens was set as the reference genus, so the coefficients for Pan, Papio, and Pongo "
        "represent their adjusted log-odds of AMTL relative to humans after accounting for age, sex, and tooth class."
    )

    for detail in genus_details:
        direction = "lower" if detail["coef"] < 0 else "higher"
        lines.append(
            f"For {detail['genus']}, the adjusted log-odds difference relative to humans was "
            f"{detail['coef']:.3f} (95% CI {detail['lower']:.3f} to {detail['upper']:.3f}), corresponding to an odds "
            f"ratio of {detail['odds_ratio']:.2f}, indicating {direction} odds of AMTL than humans."
        )

    for _, row in genus_summary.iterrows():
        lines.append(
            f"Unadjusted, {row['genus']} showed AMTL in {row['prop_amtl']:.3f} of observed sockets "
            f"({int(row['num_amtl_sockets'])} of {int(row['total_sockets'])})."
        )

    if response == "Yes":
        lines.append(
            "Because all non-human genera had significantly lower adjusted log-odds of AMTL than Homo sapiens "
            "(their 95% confidence intervals lay entirely below zero), the data support the conclusion that modern "
            "humans have higher frequencies of AMTL than Pan, Papio, and Pongo after accounting for age, sex, "
            "and tooth class."
        )
    else:
        lines.append(
            "Because at least one non-human genus did not have significantly lower adjusted log-odds of AMTL than "
            "Homo sapiens (its 95% confidence interval included zero), the data do not support the claim that modern "
            "humans have higher frequencies of AMTL than all of Pan, Papio, and Pongo after accounting for age, sex, "
            "and tooth class."
        )

    explanation = " ".join(lines)

    with open("conclusion.txt", "w") as f:
        json.dump({"response": response, "explanation": explanation}, f)


if __name__ == "__main__":
    main()

