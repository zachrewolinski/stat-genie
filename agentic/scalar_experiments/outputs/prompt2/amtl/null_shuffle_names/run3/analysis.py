import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def load_and_prepare_data(csv_path: Path, info_path: Path) -> pd.DataFrame:
    """Load AMTL data and align column semantics using info.json metadata."""
    # Load data
    df = pd.read_csv(csv_path)

    # Load metadata (not strictly required for modeling, but used to document mapping)
    with info_path.open("r") as f:
        _info = json.load(f)

    # Original columns:
    # sockets,prob_male,genus,age,pop,num_amtl,stdev_age,tooth_class,specimen
    # Based on info.json, the semantics are:
    #   sockets      -> tooth class (Anterior/Posterior/Premolar)
    #   prob_male    -> specimen identifier
    #   genus        -> number of missing teeth of that class
    #   age          -> number of observable sockets that could be scored
    #   pop          -> estimated age at death
    #   num_amtl     -> uncertainty of age at death
    #   stdev_age    -> estimated probability of being male
    #   tooth_class  -> taxonomic genus (Homo sapiens, Pan, Papio, Pongo, ...)
    #   specimen     -> region / population label
    df = df.rename(
        columns={
            "sockets": "tooth_class",
            "prob_male": "specimen_id",
            "genus": "num_missing",
            "age": "num_sockets",
            "pop": "age_at_death",
            "num_amtl": "age_uncertainty",
            "stdev_age": "prob_male",
            "tooth_class": "genus",
            "specimen": "region",
        }
    )

    # Keep only the genera relevant to the research question
    genera_of_interest = ["Homo sapiens", "Pan", "Papio", "Pongo"]
    df = df[df["genus"].isin(genera_of_interest)].copy()

    # Ensure positive socket counts
    df = df[df["num_sockets"] > 0].copy()

    # Make sure counts are integers and internally consistent
    df["num_missing"] = df["num_missing"].round().astype(int)
    df["num_sockets"] = df["num_sockets"].round().astype(int)
    df.loc[df["num_missing"] < 0, "num_missing"] = 0
    too_many_missing = df["num_missing"] > df["num_sockets"]
    if too_many_missing.any():
        df.loc[too_many_missing, "num_missing"] = df.loc[too_many_missing, "num_sockets"]

    # Indicator for modern humans vs non-human primates
    df["is_human"] = (df["genus"] == "Homo sapiens").astype(int)

    # Proportion of missing teeth per tooth class
    df["missing_prop"] = df["num_missing"] / df["num_sockets"]

    return df


def expand_to_tooth_level(df: pd.DataFrame) -> pd.DataFrame:
    """Expand aggregated counts to tooth-level binary outcomes for logistic regression."""
    records = []
    for _, row in df.iterrows():
        n_missing = int(row["num_missing"])
        n_sockets = int(row["num_sockets"])
        if n_sockets <= 0:
            continue
        n_missing = max(0, min(n_missing, n_sockets))
        n_present = n_sockets - n_missing

        # Create rows for missing teeth (amtl=1)
        if n_missing:
            missing_row = row.to_dict()
            missing_row["amtl"] = 1
            records.extend([missing_row] * n_missing)

        # Create rows for present teeth (amtl=0)
        if n_present:
            present_row = row.to_dict()
            present_row["amtl"] = 0
            records.extend([present_row] * n_present)

    return pd.DataFrame.from_records(records)


def fit_model(df_long: pd.DataFrame):
    """Fit logistic regression of AMTL on human status, age, sex, and tooth class."""
    formula = "amtl ~ is_human + age_at_death + prob_male + C(tooth_class)"
    model = smf.logit(formula=formula, data=df_long)
    result = model.fit(disp=False)
    return result


def build_conclusion(df: pd.DataFrame, result) -> dict:
    """Construct the JSON-ready conclusion based on model results."""
    coef = float(result.params["is_human"])
    pval = float(result.pvalues["is_human"])

    or_val = float(np.exp(coef))
    ci_low, ci_high = result.conf_int().loc["is_human"]
    or_ci_low = float(np.exp(ci_low))
    or_ci_high = float(np.exp(ci_high))

    # Decide Yes/No: Do humans have higher AMTL frequencies?
    if coef > 0 and pval < 0.05:
        response = "Yes"
    else:
        response = "No"

    # Map p-value and effect direction to a confidence score
    if pval < 0.001:
        base_conf = 95
    elif pval < 0.01:
        base_conf = 90
    elif pval < 0.05:
        base_conf = 80
    elif pval < 0.1:
        base_conf = 65
    else:
        base_conf = 55

    if response == "No" and coef < 0 and pval < 0.05:
        confidence = min(100, base_conf + 5)
    else:
        confidence = base_conf

    # Descriptive statistics
    n_rows = int(df.shape[0])
    n_specimens = int(df["specimen_id"].nunique())
    n_human_rows = int(df[df["is_human"] == 1].shape[0])
    n_nonhuman_rows = n_rows - n_human_rows

    human_mean = float(df[df["is_human"] == 1]["missing_prop"].mean())
    nonhuman_mean = float(df[df["is_human"] == 0]["missing_prop"].mean())

    direction_text: str
    if coef > 0 and pval < 0.05:
        direction_text = (
            "After adjusting for age at death, sex, and tooth class, "
            "modern humans show a statistically significant increase in the odds of AMTL "
            "relative to non-human primates."
        )
    elif coef < 0 and pval < 0.05:
        direction_text = (
            "After adjusting for age at death, sex, and tooth class, "
            "modern humans show a statistically significant decrease in the odds of AMTL "
            "relative to non-human primates."
        )
    else:
        direction_text = (
            "After adjusting for age at death, sex, and tooth class, "
            "the human vs. non-human difference in AMTL odds is not statistically "
            "distinguishable from zero at conventional significance levels."
        )

    explanation = (
        "I analyzed the AMTL dataset described in info.json, which contains {n_rows} "
        "tooth-class observations from {n_specimens} individual specimens across modern humans "
        "(Homo sapiens) and three non-human primate genera (Pan, Papio, Pongo). "
        "Using the metadata, I re-labeled the columns so that the model used the number of missing "
        "teeth per tooth class (num_missing), the number of observable sockets (num_sockets), "
        "estimated age at death (age_at_death), an estimated probability of being male (prob_male), "
        "and tooth class (anterior, posterior, premolar). I then expanded the aggregated counts into "
        "tooth-level binary outcomes (amtl = 1 if a tooth from that class was missing, 0 otherwise) "
        "and fit a logistic regression of AMTL on an indicator for modern humans vs. non-human primates, "
        "age at death, probability of being male, and tooth class. "
        "Unadjusted mean AMTL proportions were {human_mean:.3f} for humans and {nonhuman_mean:.3f} for "
        "non-human primates, based on {n_human_rows} human and {n_nonhuman_rows} non-human tooth-class "
        "observations. In the regression, the human indicator had an odds ratio of {or_val:.2f} "
        "with a 95% confidence interval of [{or_ci_low:.2f}, {or_ci_high:.2f}] and a p-value of "
        "{pval:.3g}. {direction_text} "
        "Based on this model, I answer the research question with a '{response}' regarding whether "
        "modern humans have higher frequencies of AMTL than non-human primates after controlling for "
        "age, sex, and tooth class."
    ).format(
        n_rows=n_rows,
        n_specimens=n_specimens,
        human_mean=human_mean,
        nonhuman_mean=nonhuman_mean,
        n_human_rows=n_human_rows,
        n_nonhuman_rows=n_nonhuman_rows,
        or_val=or_val,
        or_ci_low=or_ci_low,
        or_ci_high=or_ci_high,
        pval=pval,
        direction_text=direction_text,
        response=response,
    )

    conclusion = {
        "response": response,
        "confidence": int(confidence),
        "explanation": explanation,
    }
    return conclusion


def main() -> None:
    base_dir = Path(".")
    csv_path = base_dir / "amtl.csv"
    info_path = base_dir / "info.json"

    df = load_and_prepare_data(csv_path, info_path)
    df_long = expand_to_tooth_level(df)
    result = fit_model(df_long)
    conclusion = build_conclusion(df, result)

    with open(base_dir / "conclusion.txt", "w") as f:
        json.dump(conclusion, f)


if __name__ == "__main__":
    main()

