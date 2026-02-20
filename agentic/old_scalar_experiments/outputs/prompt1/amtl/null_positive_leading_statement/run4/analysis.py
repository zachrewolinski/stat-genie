import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def load_and_clean_data(csv_path: Path) -> pd.DataFrame:
    """Load AMTL dataset and apply basic cleaning for modeling."""
    df = pd.read_csv(csv_path)

    # Keep only the four genera of interest
    target_genera = {"Homo sapiens", "Pan", "Papio", "Pongo"}
    df = df[df["genus"].isin(target_genera)].copy()

    # Drop rows with invalid or missing counts
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["num_amtl", "sockets", "age", "prob_male", "tooth_class", "genus"])

    # Remove rows that violate the binomial constraint num_amtl <= sockets or non-positive sockets
    df = df[(df["sockets"] > 0) & (df["num_amtl"] >= 0) & (df["num_amtl"] <= df["sockets"])].copy()

    return df


def expand_to_socket_level(df: pd.DataFrame) -> pd.DataFrame:
    """
    Expand aggregated tooth-class records to one row per observable socket,
    with a binary indicator for whether that socket is missing (AMTL).
    """
    df = df.reset_index(drop=True)
    # Repeat each row by the number of sockets
    df_long = df.loc[df.index.repeat(df["sockets"])].copy()
    # Within each original record, mark the first num_amtl sockets as missing
    df_long["socket_index"] = df_long.groupby(level=0).cumcount()
    df_long["amtl"] = (df_long["socket_index"] < df_long["num_amtl"]).astype(int)
    return df_long


def fit_logistic_model(df_long: pd.DataFrame):
    """
    Fit a logistic regression at the socket level:
      logit(P(AMTL)) ~ genus + age + prob_male + tooth_class
    """
    formula = "amtl ~ C(genus, Treatment(reference='Homo sapiens')) + age + prob_male + C(tooth_class)"
    model = smf.glm(formula=formula, data=df_long, family=sm.families.Binomial())
    result = model.fit()
    return result


def compute_standardized_rates(result, df_long: pd.DataFrame):
    """
    Compute model-based, covariate-adjusted AMTL rates for each genus
    by predicting on a common covariate distribution of sockets.
    """
    genera = ["Homo sapiens", "Pan", "Papio", "Pongo"]
    rates = {}

    base_df = df_long.copy()

    for g in genera:
        tmp = base_df.copy()
        tmp["genus"] = g
        p = result.predict(tmp)
        rates[g] = float(p.mean())

    return rates


def infer_answer(result, rates):
    """
    Decide whether humans have higher AMTL frequency than
    each non-human genus after accounting for covariates.
    """
    params = result.params
    conf_int = result.conf_int()

    # Coefficients are differences in log-odds relative to Homo sapiens
    # Negative, significantly-below-zero coefficients mean the non-human genus
    # has lower AMTL than humans.
    nonhuman_terms = [
        "C(genus, Treatment(reference='Homo sapiens'))[T.Pan]",
        "C(genus, Treatment(reference='Homo sapiens'))[T.Papio]",
        "C(genus, Treatment(reference='Homo sapiens'))[T.Pongo]",
    ]

    evidence_all_lower = True
    details = []

    for term in nonhuman_terms:
        if term not in params.index:
            # If a term is absent (e.g., genus missing), we cannot claim humans are higher than that genus.
            evidence_all_lower = False
            details.append(f"Model term {term} not present; cannot compare for that genus.")
            continue

        est = params[term]
        ci_low, ci_high = conf_int.loc[term]
        # We interpret "significantly lower than Homo sapiens" as the entire CI being below 0.
        is_lower = ci_high < 0.0
        details.append(
            f"{term}: estimate={est:.3f}, 95% CI=({ci_low:.3f}, {ci_high:.3f}), "
            f"non-human lower than humans={is_lower}"
        )
        if not is_lower:
            evidence_all_lower = False

    # Also look directly at the standardized rates
    human_rate = rates.get("Homo sapiens", np.nan)
    pan_rate = rates.get("Pan", np.nan)
    papio_rate = rates.get("Papio", np.nan)
    pongo_rate = rates.get("Pongo", np.nan)

    # Humans must have the highest standardized rate to support a "Yes"
    max_rate_genus = max(rates, key=rates.get)
    humans_highest_rate = max_rate_genus == "Homo sapiens"

    response_yes = evidence_all_lower and humans_highest_rate

    return response_yes, details, {
        "Homo sapiens": human_rate,
        "Pan": pan_rate,
        "Papio": papio_rate,
        "Pongo": pongo_rate,
    }


def build_explanation(response_yes, model_details, rates, df):
    """
    Construct a concise, plain-language explanation summarizing
    the modeling approach and evidence.
    """
    n_rows = len(df)
    total_sockets = int(df["sockets"].sum())

    explanation_parts = []
    explanation_parts.append(
        f"I analyzed {n_rows} tooth-class records (covering {total_sockets} observable sockets) from Homo sapiens, Pan, Papio, and Pongo."
    )
    explanation_parts.append(
        "I expanded these records to one binary outcome per socket and modeled the probability that a socket was missing using a logistic regression with genus, age at death, probability of being male, and tooth class as predictors."
    )
    explanation_parts.append(
        "Rows with invalid counts (num_amtl > sockets or non-positive sockets) or missing covariates were removed so that the binomial model assumptions were satisfied."
    )

    # Summarize standardized rates
    human_rate = rates["Homo sapiens"]
    pan_rate = rates["Pan"]
    papio_rate = rates["Papio"]
    pongo_rate = rates["Pongo"]

    explanation_parts.append(
        "Using the fitted model, I computed covariate-adjusted AMTL rates by predicting for each genus over a common distribution of age, sex, and tooth class."
    )
    explanation_parts.append(
        f"The standardized AMTL rates were approximately {human_rate:.3f} for Homo sapiens, "
        f"{pan_rate:.3f} for Pan, {papio_rate:.3f} for Papio, and {pongo_rate:.3f} for Pongo."
    )

    # Add brief summary of coefficient evidence
    explanation_parts.append(
        "Model coefficients comparing each non-human genus to Homo sapiens in log-odds of AMTL, along with 95% confidence intervals, were:"
    )
    explanation_parts.append(" ".join(model_details))

    if response_yes:
        explanation_parts.append(
            "In this model, all non-human genera show significantly lower AMTL log-odds than Homo sapiens, and humans also have the highest covariate-adjusted AMTL rate, indicating that modern humans exhibit higher AMTL frequencies than Pan, Papio, and Pongo after accounting for age, sex, and tooth class."
        )
    else:
        explanation_parts.append(
            "In this model, at least one non-human genus does not show significantly lower AMTL log-odds than Homo sapiens and/or humans do not have the highest covariate-adjusted AMTL rate, so the data do not clearly support the claim that modern humans have higher AMTL frequencies than all three non-human genera after accounting for age, sex, and tooth class."
        )

    return " ".join(explanation_parts)


def main():
    csv_path = Path("amtl.csv")
    df = load_and_clean_data(csv_path)

    if df.empty:
        response = "No"
        explanation = (
            "After filtering to the relevant genera and removing rows with invalid or missing values, "
            "the dataset contained no usable observations, so it is not possible to determine whether "
            "modern humans have higher AMTL frequencies than non-human primates from these data."
        )
    else:
        df_long = expand_to_socket_level(df)
        result = fit_logistic_model(df_long)
        rates = compute_standardized_rates(result, df_long)
        response_yes, model_details, rate_summary = infer_answer(result, rates)
        explanation = build_explanation(response_yes, model_details, rate_summary, df)
        response = "Yes" if response_yes else "No"

    conclusion = {"response": response, "explanation": explanation}

    with open("conclusion.txt", "w") as f:
        json.dump(conclusion, f)


if __name__ == "__main__":
    main()

