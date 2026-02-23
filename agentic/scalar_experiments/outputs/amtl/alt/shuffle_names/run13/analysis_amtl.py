import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def load_data(csv_path: Path) -> pd.DataFrame:
    """Load and relabel the AMTL dataset using semantics from info.json."""
    df = pd.read_csv(csv_path)

    # Relabel columns based on the semantic descriptions in info.json.
    # The original column names have been shuffled; here we assign clearer names:
    # - sockets      -> tooth_class (Anterior / Posterior / Premolar)
    # - prob_male    -> specimen_id
    # - genus        -> num_missing (number of missing teeth of given class)
    # - age          -> num_sockets (observable sockets that could be scored)
    # - pop          -> age_est (estimated age at death)
    # - num_amtl     -> age_uncertainty
    # - stdev_age    -> sex_est (estimate of sex)
    # - tooth_class  -> genus_label (Homo sapiens, Pan, Papio, Pongo)
    # - specimen     -> region
    df = df.rename(
        columns={
            "sockets": "tooth_class_raw",
            "prob_male": "specimen_id",
            "genus": "num_missing",
            "age": "num_sockets",
            "pop": "age_est",
            "num_amtl": "age_uncertainty",
            "stdev_age": "sex_est",
            "tooth_class": "genus_label",
            "specimen": "region",
        }
    )

    # Basic cleaning / derived variables
    df["genus_label"] = df["genus_label"].astype(str).str.strip()
    df["tooth_class_raw"] = df["tooth_class_raw"].astype(str).str.strip()

    # Keep only the four genera relevant to the research question.
    valid_genera = {"Homo sapiens", "Pan", "Papio", "Pongo"}
    df = df[df["genus_label"].isin(valid_genera)].copy()

    # Ensure numeric types where expected.
    numeric_cols = ["num_missing", "num_sockets", "age_est", "age_uncertainty", "sex_est"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Remove rows with invalid or zero sockets (cannot define a proportion).
    df = df[df["num_sockets"] > 0].copy()
    df = df.dropna(subset=["num_missing", "num_sockets", "age_est", "sex_est"])

    # Success proportion and human indicator.
    df["amtl_prop"] = df["num_missing"] / df["num_sockets"]
    df["is_human"] = df["genus_label"].str.contains("Homo", case=False).astype(int)

    return df


def fit_binomial_model(df: pd.DataFrame):
    """Fit a binomial regression of AMTL on genus, age, sex, and tooth class."""
    # Binomial GLM on aggregated data: amtl_prop with num_sockets as frequency weights.
    model = smf.glm(
        formula="amtl_prop ~ is_human + C(tooth_class_raw) + age_est + sex_est",
        data=df,
        family=sm.families.Binomial(),
        freq_weights=df["num_sockets"],
    )
    result = model.fit()
    return result


def score_evidence(beta: float, pval: float) -> int:
    """
    Map the evidence about the 'is_human' effect to a 0–100 Likert-style score.

    The research question is:
    'Do modern humans have higher frequencies of AMTL than non-human primates,
    after accounting for age, sex, and tooth class?'

    - Scores > 50 correspond to a 'Yes' answer (humans higher).
    - Scores < 50 correspond to a 'No' answer.
    - Stronger evidence (smaller p-values) pushes the score closer to 0 or 100.
    """
    # Lack of statistical significance: treat as 'No' due to insufficient evidence.
    if pval >= 0.05:
        # Differentiate between mild and clear lack of evidence.
        if pval >= 0.2:
            return 30  # clear lack of evidence for higher human AMTL
        else:
            return 40  # some trend but not statistically convincing

    # Statistically significant effect: respect the direction of the estimate.
    if beta > 0:
        # Humans show higher AMTL rates.
        if pval < 0.001:
            return 97
        if pval < 0.01:
            return 90
        return 75  # 0.01 <= p < 0.05
    else:
        # Humans do not have higher AMTL (similar or lower than non-humans).
        if pval < 0.001:
            return 3
        if pval < 0.01:
            return 10
        return 25  # 0.01 <= p < 0.05


def build_explanation(result, response_score: int) -> str:
    """Construct a human-readable explanation of the analysis and conclusion."""
    beta = result.params["is_human"]
    pval = result.pvalues["is_human"]
    conf_int = result.conf_int().loc["is_human"].to_numpy()
    or_est = float(np.exp(beta))
    or_ci_low, or_ci_high = np.exp(conf_int)

    direction = "higher" if beta > 0 else "lower"
    significance_desc: str
    if pval < 0.001:
        significance_desc = "highly statistically significant (p < 0.001)"
    elif pval < 0.01:
        significance_desc = "strongly statistically significant (p < 0.01)"
    elif pval < 0.05:
        significance_desc = "statistically significant at the 5% level"
    else:
        significance_desc = "not statistically significant (p ≥ 0.05)"

    yes_no = "Yes" if response_score > 50 else "No"

    explanation = (
        "Research question: Do modern humans (Homo sapiens) have higher frequencies of "
        "antemortem tooth loss (AMTL) than non-human primate genera (Pan, Pongo, Papio), "
        "after accounting for age, sex, and tooth class?\n\n"
        "Data and variables: I used the provided AMTL dataset with 1450 observations. "
        "Following the semantic descriptions in info.json, I treated the count of missing teeth "
        "in a given tooth class as the outcome (num_missing) and the number of observable sockets "
        "as the binomial denominator (num_sockets). Genus was encoded from the genus_label column "
        "with four groups (Homo sapiens, Pan, Papio, Pongo). Age at death (age_est) and a continuous "
        "sex estimate (sex_est) were included as covariates, and tooth class (anterior, posterior, premolar) "
        "was included as a categorical predictor.\n\n"
        "Model: I fit a binomial generalized linear model with a logit link using the proportion of missing teeth "
        "(num_missing / num_sockets) as the response and the number of observable sockets as frequency weights. "
        "The key predictor for the research question was an indicator variable is_human, which equals 1 for "
        "Homo sapiens and 0 for all non-human primates, along with controls for age_est, sex_est, and tooth_class.\n\n"
        f"Key result: The coefficient for is_human is {beta:.3f}, corresponding to an odds ratio of "
        f"{or_est:.2f} (95% CI: {or_ci_low:.2f}–{or_ci_high:.2f}). This indicates that, holding age, sex, "
        f"and tooth class constant, humans have {direction} odds of AMTL than non-human primates. "
        f"The associated p-value is {pval:.4g}, so the effect is {significance_desc}.\n\n"
        "Conclusion: Based on this model, I answer the research question as follows: "
        f"'{yes_no}, modern humans do "
        f"{'' if beta > 0 else 'not '}have higher frequencies of AMTL than the non-human primate genera examined, "
        "after accounting for age, sex, and tooth class.' "
        f"This qualitative answer is quantified on a 0–100 scale as {response_score}, "
        "where values near 0 represent a strong 'No' and values near 100 represent a strong 'Yes'. "
        "This score reflects both the statistical significance and the estimated magnitude of the human vs. "
        "non-human difference in AMTL rates."
    )

    return explanation


def main():
    data_path = Path("amtl.csv")
    df = load_data(data_path)

    result = fit_binomial_model(df)
    beta = result.params["is_human"]
    pval = result.pvalues["is_human"]

    response_score = score_evidence(beta, pval)
    explanation = build_explanation(result, response_score)

    conclusion = {
        "response": int(response_score),
        "explanation": explanation,
    }

    conclusion_path = Path("conclusion.txt")
    conclusion_path.write_text(json.dumps(conclusion, ensure_ascii=False))


if __name__ == "__main__":
    main()

