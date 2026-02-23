import json
from typing import Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def load_and_prepare_data(csv_path: str) -> pd.DataFrame:
    """Load the AMTL dataset and reconstruct semantically meaningful variables."""
    df = pd.read_csv(csv_path)

    # Reconstruct semantic columns based on info.json descriptions and example rows.
    # Original columns:
    #   sockets, prob_male, genus, age, pop, num_amtl, stdev_age, tooth_class, specimen
    #
    # From the metadata and first rows we infer:
    #   - tooth_class_label: Anterior/Posterior/Premolar -> df["sockets"]
    #   - genus_label: Homo sapiens/Pan/Papio/Pongo -> df["tooth_class"]
    #   - sockets_n: number of observable sockets -> df["age"]
    #   - num_amtl_n: number of missing teeth -> df["genus"]
    #   - age_est: estimated age at death -> df["pop"]
    #   - age_uncertainty: uncertainty of age estimate -> df["num_amtl"]
    #   - prob_male_num: estimated probability of being male -> df["stdev_age"]
    #   - specimen_id: specimen identifier -> df["prob_male"]
    #   - population_label: population/region -> df["specimen"]
    df = df.copy()
    df["tooth_class_label"] = df["sockets"]
    df["genus_label"] = df["tooth_class"]
    df["sockets_n"] = df["age"].astype(float)
    df["num_amtl_n"] = df["genus"].astype(float)
    df["age_est"] = df["pop"].astype(float)
    df["age_uncertainty"] = df["num_amtl"].astype(float)
    df["prob_male_num"] = df["stdev_age"].astype(float)
    df["specimen_id"] = df["prob_male"]
    df["population_label"] = df["specimen"]

    # Derived variables for modelling
    df["is_human"] = (df["genus_label"] == "Homo sapiens").astype(int)

    # Drop rows with non-positive socket counts to avoid invalid proportions
    df = df[df["sockets_n"] > 0].copy()

    # Proportion of missing teeth in the tooth class for each specimen-row
    df["prop_amtl"] = df["num_amtl_n"] / df["sockets_n"]

    # Filter to genera of interest (Homo sapiens vs Pan, Pongo, Papio)
    genera_of_interest = ["Homo sapiens", "Pan", "Papio", "Pongo"]
    df = df[df["genus_label"].isin(genera_of_interest)].copy()

    # Remove any rows with missing values in key variables
    df = df.dropna(subset=["prop_amtl", "is_human", "age_est", "prob_male_num", "tooth_class_label", "sockets_n"])

    return df


def fit_binomial_model(df: pd.DataFrame):
    """Fit a binomial regression of AMTL proportion on genus (human vs non-human) and covariates."""
    # Use a GLM with binomial family on proportions with frequency weights equal to the number of sockets.
    formula = "prop_amtl ~ is_human + age_est + prob_male_num + C(tooth_class_label)"
    model = smf.glm(
        formula=formula,
        data=df,
        family=sm.families.Binomial(),
        freq_weights=df["sockets_n"],
    ).fit()
    return model


def summarize_group_differences(df: pd.DataFrame) -> Tuple[float, float]:
    """Compute overall AMTL proportions for humans and non-human primates."""
    grouped = (
        df.groupby("genus_label")[["num_amtl_n", "sockets_n"]]
        .sum()
        .assign(prop=lambda x: x["num_amtl_n"] / x["sockets_n"])
    )

    human_prop = float(grouped.loc["Homo sapiens", "prop"])
    nonhuman_prop = float(
        grouped.drop(index="Homo sapiens")["num_amtl_n"].sum()
        / grouped.drop(index="Homo sapiens")["sockets_n"].sum()
    )
    return human_prop, nonhuman_prop


def compute_likert_score(coef: float, pvalue: float, odds_ratio: float) -> int:
    """Map effect direction, significance, and magnitude to a 0–100 Likert score."""
    # Default neutral score if model is degenerate
    if not np.isfinite(coef) or not np.isfinite(pvalue):
        return 50

    # Base score from direction and significance
    if coef > 0:
        if pvalue < 0.001:
            base = 85
        elif pvalue < 0.01:
            base = 75
        elif pvalue < 0.05:
            base = 65
        elif pvalue < 0.1:
            base = 55
        else:
            base = 50
    else:
        if pvalue < 0.001:
            base = 15
        elif pvalue < 0.01:
            base = 25
        elif pvalue < 0.05:
            base = 35
        elif pvalue < 0.1:
            base = 45
        else:
            base = 50

    # Adjust for effect size magnitude (odds ratio)
    # Stronger deviations from OR=1 push the score further from 50.
    if np.isfinite(odds_ratio) and odds_ratio > 0:
        log_or = abs(np.log(odds_ratio))
        # Scale: small (~0), moderate (~0.5), large (~1+)
        delta = int(min(15, max(0, log_or * 10)))
        if coef > 0:
            base += delta
        else:
            base -= delta

    return int(max(0, min(100, base)))


def build_explanation(
    human_prop: float,
    nonhuman_prop: float,
    coef: float,
    pvalue: float,
    odds_ratio: float,
    likert_score: int,
) -> str:
    """Construct a narrative explanation of the analysis and findings."""
    direction = "higher" if coef > 0 else "lower"
    yes_no = "Yes" if likert_score >= 50 and coef > 0 else "No"

    explanation = (
        "Research question: Do modern humans (Homo sapiens) have higher frequencies of antemortem tooth loss (AMTL) "
        "than non-human primate genera (Pan, Pongo, Papio) after accounting for age, sex, and tooth class?\n\n"
        "Data and variables: Each row in the dataset represents a specimen–tooth-class combination. For each row, I "
        "reconstructed the number of observable tooth sockets in that class (sockets_n), the number of missing teeth "
        "in that class (num_amtl_n), the estimated age at death (age_est), an estimate of sex via the probability of "
        "being male (prob_male_num), and categorical indicators for tooth class (anterior, posterior, premolar) and "
        "genus (Homo sapiens, Pan, Papio, Pongo). The AMTL outcome is the proportion of missing teeth in a class "
        "(num_amtl_n / sockets_n).\n\n"
        f"Descriptive comparison: Aggregating across all tooth classes and specimens, modern humans show an overall "
        f"AMTL proportion of approximately {human_prop:.3f}, whereas non-human primates combined show an AMTL "
        f"proportion of approximately {nonhuman_prop:.3f}. This suggests that humans have {direction} raw AMTL "
        "frequencies before statistical adjustment.\n\n"
        "Statistical model: To formally test the research question while adjusting for covariates, I fit a binomial "
        "generalized linear model (logit link) to the AMTL proportion, using the number of observable sockets as "
        "binomial weights. The model included an indicator for modern humans versus non-human primates, continuous "
        "age at death, the probability of being male, and categorical tooth-class effects. This directly estimates "
        "the difference in AMTL frequency attributable to being human, conditional on age, sex, and tooth class.\n\n"
        f"Human effect: In this model, the coefficient for the human indicator corresponds to a log-odds difference "
        f"of {coef:.3f}, which translates to an odds ratio of about {odds_ratio:.2f} for AMTL in humans compared to "
        "non-human primates, holding age, sex, and tooth class constant. The associated p-value for this effect is "
        f"{pvalue:.3g}, indicating "
    )

    if pvalue < 0.001:
        explanation += "very strong statistical evidence that the human AMTL frequency differs from that of non-human primates.\n\n"
    elif pvalue < 0.01:
        explanation += "strong statistical evidence for a difference in AMTL frequency between humans and non-human primates.\n\n"
    elif pvalue < 0.05:
        explanation += "moderate statistical evidence for a difference in AMTL frequency between humans and non-human primates.\n\n"
    elif pvalue < 0.1:
        explanation += "weak statistical evidence for a difference; the direction of effect is suggestive but not definitive.\n\n"
    else:
        explanation += "little statistical evidence that humans differ from non-human primates in AMTL frequency after adjustment.\n\n"

    explanation += (
        f"Conclusion on the research question: {yes_no}. Based on the direction and significance of the human "
        f"coefficient, together with the magnitude of the estimated odds ratio, I translate this into a Likert-scale "
        f"response of {likert_score} on a 0–100 scale, where 0 represents a strong 'No' and 100 a strong 'Yes'. "
        "Higher scores indicate stronger support for the claim that modern humans have higher AMTL frequencies than "
        "non-human primates after controlling for age, sex, and tooth class, while lower scores indicate evidence "
        "against that claim."
    )

    return explanation


def main() -> None:
    df = load_and_prepare_data("amtl.csv")

    if df.empty:
        # Degenerate safeguard: if something went wrong and there is no data, return neutral.
        conclusion = {
            "response": 50,
            "explanation": (
                "The prepared dataset was empty after filtering, so it was not possible to assess whether modern "
                "humans have higher AMTL frequencies than non-human primates. A neutral score of 50 is returned."
            ),
        }
        with open("conclusion.txt", "w") as f:
            json.dump(conclusion, f)
        return

    model = fit_binomial_model(df)
    human_coef = float(model.params.get("is_human", np.nan))
    human_pvalue = float(model.pvalues.get("is_human", np.nan))
    human_or = float(np.exp(human_coef)) if np.isfinite(human_coef) else float("nan")

    human_prop, nonhuman_prop = summarize_group_differences(df)

    likert_score = compute_likert_score(human_coef, human_pvalue, human_or)
    explanation = build_explanation(
        human_prop=human_prop,
        nonhuman_prop=nonhuman_prop,
        coef=human_coef,
        pvalue=human_pvalue,
        odds_ratio=human_or,
        likert_score=likert_score,
    )

    conclusion = {
        "response": likert_score,
        "explanation": explanation,
    }

    with open("conclusion.txt", "w") as f:
        json.dump(conclusion, f)


if __name__ == "__main__":
    main()

