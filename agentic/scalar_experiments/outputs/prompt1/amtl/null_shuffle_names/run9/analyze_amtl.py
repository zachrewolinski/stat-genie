import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def load_and_prepare_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    # Map the documented semantics from info.json onto clearer column names.
    # See info.json: the original headers are somewhat misaligned with their meanings.
    df = df.rename(
        columns={
            "sockets": "tooth_region",  # Anterior / Posterior / Premolar
            "prob_male": "specimen_id",  # Unique specimen identifier
            "genus": "num_missing",  # Number of missing teeth of given class
            "age": "num_sockets",  # Number of observable sockets
            "pop": "age_at_death",  # Estimated age at death
            "num_amtl": "age_sd",  # Uncertainty in age at death
            "stdev_age": "prob_male",  # Estimate of sex (probability specimen is male)
            "tooth_class": "genus",  # Taxonomic genus (Homo sapiens, Pan, Papio, Pongo)
            "specimen": "region",  # Geographic/ethnographic region
        }
    )

    # Basic cleaning and sanity checks.
    df = df.dropna(
        subset=["num_missing", "num_sockets", "age_at_death", "prob_male", "tooth_region", "genus"]
    )

    # Ensure integer counts for missing teeth and observable sockets.
    df["num_missing"] = df["num_missing"].round().astype(int)
    df["num_sockets"] = df["num_sockets"].round().astype(int)

    # Keep only rows with feasible counts.
    df = df[(df["num_sockets"] > 0) & (df["num_missing"] >= 0) & (df["num_missing"] <= df["num_sockets"])]

    # Create human vs non-human indicator based on genus.
    df["is_human"] = (df["genus"] == "Homo sapiens").astype(int)

    # We only keep genera relevant to the research question.
    valid_genera = {"Homo sapiens", "Pan", "Papio", "Pongo"}
    df = df[df["genus"].isin(valid_genera)].copy()

    return df


def expand_to_tooth_level(df: pd.DataFrame) -> pd.DataFrame:
    """
    Expand each row with counts of missing vs present teeth into per-tooth binary rows.
    This allows fitting a standard logistic regression with a Bernoulli outcome.
    """
    records = []

    for _, row in df.iterrows():
        n_sockets = int(row["num_sockets"])
        n_missing = int(row["num_missing"])
        n_missing = max(0, min(n_missing, n_sockets))

        # 1 = missing tooth, 0 = present tooth
        outcomes = np.array([1] * n_missing + [0] * (n_sockets - n_missing), dtype=int)

        for y in outcomes:
            rec = {
                "missing": y,
                "is_human": row["is_human"],
                "age_at_death": row["age_at_death"],
                "prob_male": row["prob_male"],
                "tooth_region": row["tooth_region"],
            }
            records.append(rec)

    expanded = pd.DataFrame.from_records(records)

    # Drop any rows with missing covariates (should be rare after earlier cleaning).
    expanded = expanded.dropna(subset=["missing", "is_human", "age_at_death", "prob_male", "tooth_region"])

    return expanded


def fit_logistic_model(expanded: pd.DataFrame):
    """
    Fit a logistic regression for tooth loss (missing vs present) with
    human vs non-human indicator, age at death, sex estimate, and tooth region.
    """
    # Scale age to decades to keep coefficients on a reasonable scale.
    expanded = expanded.copy()
    expanded["age_decades"] = expanded["age_at_death"] / 10.0

    # prob_male is already between 0 and 1.
    formula = "missing ~ is_human + age_decades + prob_male + C(tooth_region)"

    model = smf.logit(formula=formula, data=expanded)
    result = model.fit(disp=False)
    return result


def interpret_result(result) -> dict:
    """
    Interpret the human vs non-human effect from the fitted model.
    Return a dictionary with a Yes/No response and a textual explanation.
    """
    params = result.params
    b_human = params.get("is_human", np.nan)

    pvalues = result.pvalues
    p_human = pvalues.get("is_human", np.nan)

    # Compute odds ratio and a 95% Wald confidence interval for the human effect.
    conf_int = result.conf_int().loc["is_human"]
    ci_lower, ci_upper = conf_int.iloc[0], conf_int.iloc[1]
    odds_ratio = float(np.exp(b_human))
    or_ci_lower = float(np.exp(ci_lower))
    or_ci_upper = float(np.exp(ci_upper))

    # Decide on Yes/No based on sign and statistical significance.
    # We consider evidence for higher human AMTL if:
    #   - the estimated effect is positive
    #   - and the 95% CI for the odds ratio lies entirely above 1
    #     (equivalently, CI for the coefficient above 0).
    if (b_human > 0) and (ci_lower > 0):
        response = "Yes"
    else:
        response = "No"

    explanation_lines = []
    explanation_lines.append(
        "I fit a logistic regression where each tooth was treated as present (0) or missing (1), "
        "using counts of missing and observable sockets for each specimen and tooth region."
    )
    explanation_lines.append(
        "The model included a human-vs-non-human indicator, age at death (in decades), "
        "a continuous estimate of sex (probability male), and categorical tooth region "
        "(anterior, posterior, premolar) as predictors."
    )
    explanation_lines.append(
        f"The estimated log-odds coefficient for humans relative to non-human primates was {b_human:.3f}, "
        f"corresponding to an odds ratio of {odds_ratio:.2f} "
        f"with a 95% confidence interval from {or_ci_lower:.2f} to {or_ci_upper:.2f}."
    )

    if response == "Yes":
        explanation_lines.append(
            "Because the human effect is positive and its 95% confidence interval lies entirely above 1 on the odds ratio scale, "
            "the model indicates that modern humans have higher frequencies of antemortem tooth loss than the non-human primate genera "
            "after accounting for age, sex, and tooth region."
        )
    else:
        explanation_lines.append(
            "Because the 95% confidence interval for the human effect includes no increase (odds ratio of 1), "
            "the model does not provide strong evidence that modern humans have higher frequencies of antemortem tooth loss "
            "than the non-human primate genera after accounting for age, sex, and tooth region."
        )

    if not np.isnan(p_human):
        explanation_lines.append(
            f"The two-sided p-value for the human effect was {p_human:.3g}, "
            "which quantifies the strength of evidence for a difference in AMTL between humans and non-human primates."
        )

    explanation = " ".join(explanation_lines)
    return {"response": response, "explanation": explanation}


def main():
    csv_path = Path("amtl.csv")
    df = load_and_prepare_data(csv_path)
    expanded = expand_to_tooth_level(df)

    if expanded.empty:
        result = {
            "response": "No",
            "explanation": (
                "After attempting to prepare and analyze the AMTL dataset, no valid observations remained "
                "for modeling (e.g., due to missing or inconsistent counts). "
                "Without analyzable data, I cannot provide evidence that humans have higher AMTL frequencies "
                "than non-human primates after accounting for age, sex, and tooth class."
            ),
        }
    else:
        model_result = fit_logistic_model(expanded)
        result = interpret_result(model_result)

    conclusion_path = Path("conclusion.txt")
    conclusion_path.write_text(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()

