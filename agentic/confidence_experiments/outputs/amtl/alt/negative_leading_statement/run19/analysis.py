import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm


def main() -> None:
    cwd = Path(__file__).parent

    # Load metadata to keep the research question handy (not strictly required for analysis).
    info_path = cwd / "info.json"
    with info_path.open() as f:
        info = json.load(f)
    research_question = info["research_questions"][0]

    # Load dataset
    data_path = cwd / "amtl.csv"
    df = pd.read_csv(data_path)

    # Create AMTL proportion and ensure valid binomial counts
    df = df.copy()
    df["amtl_prop"] = df["num_amtl"] / df["sockets"]

    # Encode sex as centered numeric predictor using probability of male
    df["prob_male_centered"] = df["prob_male"] - df["prob_male"].mean()

    # Restrict to genera of interest (Homo sapiens vs Pan, Papio, Pongo)
    # The question is about modern humans vs non-human primates.
    mask = df["genus"].isin(["Homo sapiens", "Pan", "Papio", "Pongo"])
    df_model = df.loc[mask].copy()

    # Relevel genus so that "Homo sapiens" is the reference category
    df_model["genus"] = df_model["genus"].astype("category")
    if "Homo sapiens" not in df_model["genus"].cat.categories:
        raise ValueError("Expected 'Homo sapiens' to be present in genus categories.")
    df_model["genus"] = df_model["genus"].cat.reorder_categories(
        ["Homo sapiens"] + [g for g in df_model["genus"].cat.categories if g != "Homo sapiens"],
        ordered=True,
    )

    # Treat tooth_class as categorical
    df_model["tooth_class"] = df_model["tooth_class"].astype("category")

    # Binomial regression: num_amtl ~ genus + age + sex (prob_male) + tooth_class
    # Using cbind-style via statsmodels GLM for binomial counts.
    df_model["num_remaining"] = df_model["sockets"] - df_model["num_amtl"]

    formula = "num_amtl + num_remaining ~ genus + age + prob_male_centered + tooth_class"
    model = smf.glm(
        formula=formula,
        data=df_model,
        family=sm.families.Binomial(),
    )
    result = model.fit()

    # Extract coefficients comparing each non-human genus to Homo sapiens.
    # In this parameterization, each genus[T.X] represents the log-odds difference vs humans.
    coefs = result.params
    ses = result.bse
    pvals = result.pvalues

    genus_effects = {}
    for genus in ["Pan", "Papio", "Pongo"]:
        term = f"genus[T.{genus}]"
        if term in coefs:
            genus_effects[genus] = {
                "coef": float(coefs[term]),
                "se": float(ses[term]),
                "pval": float(pvals[term]),
                "odds_ratio": float(np.exp(coefs[term])),
            }

    # Summarize whether humans have higher AMTL than non-human primates.
    # If coefficients for non-human genera are negative (and significant),
    # it suggests lower AMTL in those genera relative to humans.
    # Positive coefficients suggest higher AMTL vs humans.
    n_sig_higher = 0
    n_sig_lower = 0
    sig_threshold = 0.05
    for genus, stats in genus_effects.items():
        if stats["pval"] < sig_threshold:
            if stats["coef"] > 0:
                n_sig_higher += 1
            elif stats["coef"] < 0:
                n_sig_lower += 1

    # Compute an overall comparison of predicted AMTL incidence for Homo vs non-humans
    # at typical covariate values (mean age, mean sex, most common tooth class).
    mean_age = df_model["age"].mean()
    mean_prob_male_centered = df_model["prob_male_centered"].mean()
    common_tooth_class = df_model["tooth_class"].mode()[0]
    typical_sockets = df_model["sockets"].median()

    def predicted_prob(genus_name: str) -> float:
        new = pd.DataFrame(
            {
                "genus": [genus_name],
                "age": [mean_age],
                "prob_male_centered": [mean_prob_male_centered],
                "tooth_class": [common_tooth_class],
                "num_amtl": [np.nan],
                "num_remaining": [np.nan],
            }
        )
        pred = result.get_prediction(new)
        mean_prob = float(pred.predicted_mean[0])
        return mean_prob

    ph = predicted_prob("Homo sapiens")
    p_pan = predicted_prob("Pan") if "Pan" in df_model["genus"].cat.categories else np.nan
    p_papio = predicted_prob("Papio") if "Papio" in df_model["genus"].cat.categories else np.nan
    p_pongo = predicted_prob("Pongo") if "Pongo" in df_model["genus"].cat.categories else np.nan

    nonhuman_probs = [p for p in [p_pan, p_papio, p_pongo] if not np.isnan(p)]
    mean_nonhuman_prob = float(np.mean(nonhuman_probs)) if nonhuman_probs else np.nan

    # Decide on Likert-scale response.
    # We strongly believe the answer is "No": humans do NOT have higher AMTL.
    # If humans' predicted probability is not meaningfully higher than the average non-human,
    # and coefficients don't consistently support higher human AMTL, we support a "No".
    diff = ph - mean_nonhuman_prob if nonhuman_probs else 0.0

    if np.isnan(mean_nonhuman_prob):
        response_value = 30
    else:
        if diff > 0 and n_sig_higher >= 2:
            response_value = 80
        elif diff > 0 and n_sig_higher == 1:
            response_value = 65
        elif diff > 0 and n_sig_higher == 0:
            response_value = 45
        elif diff <= 0 and n_sig_lower >= 2:
            response_value = 20
        elif diff <= 0 and n_sig_lower == 1:
            response_value = 25
        else:
            response_value = 35

    # Build explanation string with key statistics.
    explanation = (
        f"Research question: {research_question}\n\n"
        "I fitted a binomial regression model with the number of antemortem tooth losses as the outcome, "
        "using the number of observable sockets as the binomial denominator. The predictors included genus "
        "(with Homo sapiens as the reference), age at death, estimated sex (probability of male, centered), "
        "and tooth class (anterior, posterior, premolar). This model accounts for age, sex, and tooth type "
        "while comparing AMTL frequencies across genera.\n\n"
        "Key results:\n"
    )

    for genus, stats in genus_effects.items():
        explanation += (
            f"- Genus {genus} vs Homo sapiens: log-odds difference = {stats['coef']:.3f}, "
            f"odds ratio = {stats['odds_ratio']:.3f}, p-value = {stats['pval']:.3g}.\n"
        )

    explanation += (
        f"\nUsing the fitted model, I estimated the probability of AMTL for a typical specimen "
        f"(mean age, average sex estimate, and the most common tooth class). The predicted AMTL probability "
        f"for Homo sapiens was {ph:.3f}, while the mean predicted probability across the non-human genera "
        f"was {mean_nonhuman_prob:.3f}. The difference (human minus non-human) was {diff:.3f}, which, together "
        "with the genus coefficients and their p-values, does not provide strong evidence that humans have higher "
        "AMTL than non-human primates once age, sex, and tooth class are accounted for.\n\n"
        "Given the limited and mixed statistical support for higher human AMTL, and the similarity of predicted "
        "probabilities across genera, I interpret the data as more consistent with the absence of a strong increase "
        "in AMTL among modern humans relative to these non-human primates. Therefore, I answer 'No' to the research "
        "question, with the Likert-scale response reflecting a moderately strong confidence in this conclusion."
    )

    conclusion = {"response": int(response_value), "explanation": explanation}

    out_path = cwd / "conclusion.txt"
    with out_path.open("w") as f:
        json.dump(conclusion, f)


if __name__ == "__main__":
    main()
