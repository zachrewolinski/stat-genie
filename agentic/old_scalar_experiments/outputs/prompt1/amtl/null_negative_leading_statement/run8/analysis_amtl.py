import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def load_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # Drop any rows with non-positive socket counts, just in case
    df = df[df["sockets"] > 0].copy()
    # Proportion of missing teeth for binomial modeling
    df["amtl_rate"] = df["num_amtl"] / df["sockets"]
    return df


def fit_binomial_model(df: pd.DataFrame):
    # Binomial regression of AMTL rate with sockets as binomial trials
    # Homo sapiens is set as the reference genus
    formula = (
        "amtl_rate ~ C(genus, Treatment(reference='Homo sapiens'))"
        " + age + prob_male + C(tooth_class)"
    )
    model = smf.glm(
        formula=formula,
        data=df,
        family=sm.families.Binomial(),
        freq_weights=df["sockets"],
    )
    result = model.fit()
    return result


def compute_descriptive_rates(df: pd.DataFrame):
    grouped = df.groupby("genus").agg(
        total_amtl=("num_amtl", "sum"),
        total_sockets=("sockets", "sum"),
    )
    grouped["amtl_rate"] = grouped["total_amtl"] / grouped["total_sockets"]
    return grouped


def compute_marginal_predicted_rates(df: pd.DataFrame, result) -> pd.Series:
    """
    Compute marginal predicted AMTL probabilities for each genus by
    setting genus to each level in turn while holding all other
    covariates at their observed values, then averaging predictions
    weighted by sockets.
    """
    genera = sorted(df["genus"].unique())
    marginal_rates = {}
    for g in genera:
        df_copy = df.copy()
        df_copy["genus"] = g
        preds = result.predict(df_copy)
        # Weighted average per socket
        marginal_rates[g] = float((preds * df_copy["sockets"]).sum() / df_copy["sockets"].sum())
    return pd.Series(marginal_rates)


def assess_human_higher(result) -> bool:
    """
    Determine whether modern humans (Homo sapiens) have significantly
    higher AMTL frequency than each non-human genus after adjustment.

    Because Homo sapiens is the reference level, coefficients for
    C(genus)[T.G] represent log-odds differences (G minus Homo).
    Homo has higher AMTL than genus G if this coefficient is negative.
    We require the 95% CI upper bound for each such coefficient to be < 0
    to consider the evidence strong that Homo > all non-human genera.
    """
    params = result.params
    conf = result.conf_int()
    genus_coef_rows = [
        name
        for name in params.index
        if name.startswith("C(genus") and "Homo sapiens" not in name
    ]
    if not genus_coef_rows:
        # If for some reason there are no non-human genera, we cannot support the claim
        return False

    # Upper bound of CI < 0 => coefficient significantly negative => Homo higher
    upper_bounds = conf.loc[genus_coef_rows, 1]
    return bool((upper_bounds < 0).all())


def build_explanation(
    df: pd.DataFrame,
    descr: pd.DataFrame,
    marginal: pd.Series,
    result,
    humans_higher: bool,
) -> str:
    # Basic descriptive statistics
    n_rows = len(df)
    genera = sorted(df["genus"].unique())

    descr_lines = []
    for g in genera:
        row = descr.loc[g]
        descr_lines.append(
            f"{g}: {row['total_amtl']} missing teeth out of "
            f"{row['total_sockets']} sockets "
            f"(raw AMTL rate {row['amtl_rate']:.3f})"
        )

    marginal_lines = []
    for g in genera:
        marginal_lines.append(f"{g}: adjusted AMTL probability {marginal[g]:.3f}")

    # Summarize genus effects from model
    params = result.params
    pvalues = result.pvalues
    genus_effect_lines = []
    for name, coef in params.items():
        if name.startswith("C(genus"):
            genus_effect_lines.append(
                f"{name}: coefficient {coef:.3f}, p={pvalues[name]:.3g}"
            )

    conclusion_sentence = (
        "Based on the adjusted binomial regression model, there is strong "
        "evidence that modern humans have higher AMTL frequencies than all "
        "non-human primate genera examined."
        if humans_higher
        else
        "Based on the adjusted binomial regression model, there is not clear "
        "evidence that modern humans have higher AMTL frequencies than all "
        "non-human primate genera; if anything, some non-human genera show "
        "similar or higher AMTL rates."
    )

    explanation = (
        "I analyzed the antemortem tooth loss (AMTL) dataset using binomial "
        "regression to compare modern humans (Homo sapiens) to non-human "
        "primates (Pan, Papio, Pongo) while adjusting for age, sex, and tooth "
        "class.\n"
        f"The dataset contains {n_rows} observations with genus categories: "
        f"{', '.join(genera)}.\n"
        "Raw AMTL rates by genus (total missing teeth / total sockets) are:\n"
        + "\n".join(descr_lines)
        + "\n\nAdjusted AMTL probabilities from the binomial regression "
        "(holding covariates constant across genera) are:\n"
        + "\n".join(marginal_lines)
        + "\n\nKey genus effects from the regression (Homo sapiens as the "
        "reference category) are:\n"
        + "\n".join(genus_effect_lines)
        + "\n\n"
        + conclusion_sentence
    )
    return explanation


def main():
    csv_path = Path("amtl.csv")
    df = load_data(csv_path)

    descr = compute_descriptive_rates(df)
    result = fit_binomial_model(df)
    marginal = compute_marginal_predicted_rates(df, result)
    humans_higher = assess_human_higher(result)

    explanation = build_explanation(df, descr, marginal, result, humans_higher)
    response_value = "Yes" if humans_higher else "No"

    conclusion = {
        "response": response_value,
        "explanation": explanation,
    }

    out_path = Path("conclusion.txt")
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(conclusion, f, ensure_ascii=False)


if __name__ == "__main__":
    main()

