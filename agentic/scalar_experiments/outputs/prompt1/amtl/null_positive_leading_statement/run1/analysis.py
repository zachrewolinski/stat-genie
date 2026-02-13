import json
from pathlib import Path

import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


DATA_FILE = Path("amtl.csv")
INFO_FILE = Path("info.json")
OUTPUT_FILE = Path("conclusion.txt")


def main() -> None:
    with INFO_FILE.open() as f:
        info = json.load(f)

    research_questions = info.get("research_questions", [])
    research_question = research_questions[0] if research_questions else ""

    df = pd.read_csv(DATA_FILE)

    # Basic cleaning: ensure key variables are present and sockets are positive.
    df = df.dropna(subset=["num_amtl", "sockets", "age", "prob_male", "genus", "tooth_class"])
    df = df[df["sockets"] > 0].copy()

    # Expand to a tooth-level dataset so that each row represents a single tooth
    # socket with a binary AMTL outcome (1 = missing, 0 = present). This avoids
    # numerical issues with aggregated binomial fitting and is equivalent to
    # modeling counts with a binomial model.
    records = []
    for _, row in df.iterrows():
        n_sockets = int(row["sockets"])
        n_amtl = int(row["num_amtl"])
        if n_sockets <= 0:
            continue
        # Clamp AMTL counts to the valid range [0, n_sockets].
        n_amtl = max(0, min(n_amtl, n_sockets))
        base = row.to_dict()
        # Remove count fields from the base covariate set.
        base.pop("num_amtl", None)
        base.pop("sockets", None)
        outcomes = [1] * n_amtl + [0] * (n_sockets - n_amtl)
        for y in outcomes:
            rec = base.copy()
            rec["amtl"] = y
            records.append(rec)

    df_long = pd.DataFrame.from_records(records)

    # Fit logistic regression on the tooth-level data:
    # Response: amtl (1 if socket is missing, 0 otherwise).
    # Predictors: genus (Homo sapiens as reference), age, sex proxy (prob_male), tooth class.
    formula = (
        "amtl ~ "
        "C(genus, Treatment(reference='Homo sapiens')) + "
        "age + prob_male + C(tooth_class)"
    )

    model = smf.logit(
        formula=formula,
        data=df_long,
    ).fit(disp=False)

    # Standardized, model-based AMTL probabilities for each genus:
    # For each genus, predict AMTL probability for every tooth-level observation
    # while holding age, sex, and tooth class at their observed values but
    # forcing genus to the target level, then average across teeth.
    genuses = sorted(df_long["genus"].unique())
    genus_pred = {}

    for g in genuses:
        df_tmp = df_long.copy()
        df_tmp["genus"] = g
        pred = model.predict(df_tmp)
        avg_p = float(pred.mean())
        genus_pred[g] = avg_p

    # Evaluate hypothesis: Homo sapiens vs non-human genera.
    if "Homo sapiens" not in genus_pred:
        raise ValueError("Expected 'Homo sapiens' genus in data.")

    homo_p = genus_pred["Homo sapiens"]
    others = [g for g in genuses if g != "Homo sapiens"]

    # 1) Check whether Homo sapiens has the highest predicted AMTL probability.
    higher_than_all = all(homo_p > genus_pred[g] for g in others)

    # 2) Check whether non-human genera have significantly lower AMTL than humans,
    #    based on genus coefficients (Homo sapiens is the reference).
    sig_higher = True
    genus_effect_summaries = []

    for g in others:
        param_name = f"C(genus, Treatment(reference='Homo sapiens'))[T.{g}]"
        if param_name in model.params:
            est = float(model.params[param_name])
            pval = float(model.pvalues[param_name])
            genus_effect_summaries.append(
                f"{g} vs Homo sapiens: log-odds difference {est:.3f}, p={pval:.3g}"
            )
            # For humans to have higher AMTL, these differences should be negative
            # and statistically significant (p < 0.05).
            if not (est < 0 and pval < 0.05):
                sig_higher = False
        else:
            # If a contrast is unavailable, be conservative.
            sig_higher = False

    answer_yes = higher_than_all and sig_higher
    response = "Yes" if answer_yes else "No"

    # Build explanation text.
    explanation_parts = []

    if research_question:
        explanation_parts.append(f"Research question: {research_question}")
    else:
        explanation_parts.append(
            "Research question: Do modern humans (Homo sapiens) have higher "
            "frequencies of antemortem tooth loss (AMTL) than non-human primate "
            "genera (Pan, Papio, Pongo) after controlling for age, sex, and tooth class?"
        )

    explanation_parts.append(
        "I analyzed the dataset using a logistic regression on an expanded "
        "tooth-level dataset, where each observable socket was represented as a "
        "single row with a binary AMTL outcome (1 = missing, 0 = present). "
        "Predictors included genus (with Homo sapiens as the reference category), "
        "estimated age at death, the probability of being male (prob_male), and "
        "tooth class (anterior, premolar, posterior). This model accounts for "
        "age, sex, and tooth class while estimating genus differences in AMTL."
    )

    genus_summaries = ", ".join(
        f"{g}: {genus_pred[g]:.3f}" for g in genuses
    )
    explanation_parts.append(
        "From the fitted model, I computed standardized, model-based AMTL "
        f"probabilities for each genus by predicting AMTL for every observation "
        f"while varying only genus. The average predicted AMTL probabilities per "
        f"tooth socket were: {genus_summaries}."
    )

    if genus_effect_summaries:
        explanation_parts.append(
            "Genus effects from the logistic regression (negative values indicate "
            "lower AMTL than humans; p-values are two-sided tests of no difference) "
            + "; ".join(genus_effect_summaries)
            + "."
        )

    if answer_yes:
        explanation_parts.append(
            "Homo sapiens shows the highest model-based AMTL probability, and all "
            "non-human genera have significantly lower AMTL (negative log-odds "
            "differences with p < 0.05) after adjusting for age, sex, and tooth "
            "class. Therefore, the data support the conclusion that modern humans "
            "have higher AMTL frequencies than the non-human primate genera in this sample."
        )
    else:
        explanation_parts.append(
            "Although the model adjusts for age, sex, and tooth class, the estimated "
            "genus effects do not consistently show that Homo sapiens has both the "
            "highest AMTL probability and significantly higher AMTL than each "
            "non-human genus (i.e., not all non-human genera have significantly "
            "negative log-odds differences relative to humans at p < 0.05). "
            "Accordingly, I do not conclude that modern humans have higher AMTL "
            "frequencies than all non-human primate genera based on this dataset."
        )

    explanation = " ".join(explanation_parts)

    result = {
        "response": response,
        "explanation": explanation,
    }

    with OUTPUT_FILE.open("w") as f:
        json.dump(result, f)


if __name__ == "__main__":
    main()
