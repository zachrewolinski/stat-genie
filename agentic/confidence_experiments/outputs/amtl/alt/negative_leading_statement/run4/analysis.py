import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from patsy import dmatrix


DESIGN_FORMULA = (
    "C(genus, Treatment(reference='Homo sapiens')) "
    "+ age + prob_male + C(tooth_class)"
)


def load_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # Basic sanity filters
    df = df[df["sockets"] > 0].copy()
    df = df.dropna(subset=["num_amtl", "sockets", "age", "prob_male", "genus", "tooth_class"])
    df["genus"] = df["genus"].astype("category")
    df["tooth_class"] = df["tooth_class"].astype("category")
    return df


def fit_binomial_model(df: pd.DataFrame):
    """
    Binomial GLM using a two-column response (successes, failures):
    endog = [num_amtl, sockets - num_amtl]
    with a design matrix for predictors created via patsy.dmatrix.
    """
    # Design matrix with Homo sapiens as the reference genus
    X = dmatrix(DESIGN_FORMULA, df, return_type="dataframe")
    y = np.column_stack([df["num_amtl"].to_numpy(), (df["sockets"] - df["num_amtl"]).to_numpy()])

    model = sm.GLM(y, X, family=sm.families.Binomial())
    result = model.fit()
    return result


def summarize_genus_effects(result, df: pd.DataFrame):
    # Extract genus coefficients (differences vs Homo sapiens)
    genus_params = {
        name: result.params[name]
        for name in result.params.index
        if "C(genus" in name
    }
    genus_pvalues = {
        name: result.pvalues[name]
        for name in genus_params.keys()
    }

    # Predicted probabilities per genus at typical covariate values
    mean_age = float(df["age"].mean())
    mean_prob_male = float(df["prob_male"].mean())

    # Use the most common tooth class as reference profile
    ref_tooth_class = df["tooth_class"].mode().iat[0]

    genera = sorted(df["genus"].unique())
    pred_df = pd.DataFrame(
        {
            "genus": genera,
            "age": [mean_age] * len(genera),
            "prob_male": [mean_prob_male] * len(genera),
            "tooth_class": [ref_tooth_class] * len(genera),
        }
    )
    # Ensure categorical dtypes align
    pred_df["genus"] = pred_df["genus"].astype(df["genus"].dtype)
    pred_df["tooth_class"] = pred_df["tooth_class"].astype(df["tooth_class"].dtype)

    X_pred = dmatrix(DESIGN_FORMULA, pred_df, return_type="dataframe")
    predicted_probs = result.predict(X_pred)
    genus_pred = {
        str(genus): float(prob)
        for genus, prob in zip(pred_df["genus"], predicted_probs)
    }

    return genus_params, genus_pvalues, genus_pred, ref_tooth_class


def build_conclusion(
    genus_params,
    genus_pvalues,
    genus_pred,
    ref_tooth_class,
):
    """
    Decide on Yes/No and Likert score based on:
    - Direction and significance of genus coefficients (Pan, Pongo, Papio vs Homo sapiens)
    - Predicted AMTL probabilities per genus at typical covariate values.
    """
    # Identify non-human genera explicitly
    non_human_genera = [g for g in genus_pred.keys() if g != "Homo sapiens"]

    # Map from statsmodels coefficient names to genera
    coef_by_genus = {}
    pval_by_genus = {}
    for name, coef in genus_params.items():
        # Example name: C(genus, Treatment(reference='Homo sapiens'))[T.Pan]
        if "[T." in name and "]" in name:
            genus = name.split("[T.", 1)[1].split("]", 1)[0]
            coef_by_genus[genus] = float(coef)
            pval_by_genus[genus] = float(genus_pvalues[name])

    # Determine evidence pattern for each non-human genus
    evidence = {}
    for genus in non_human_genera:
        coef = coef_by_genus.get(genus, np.nan)
        pval = pval_by_genus.get(genus, np.nan)
        evidence[genus] = {"coef": coef, "pval": pval}

    # Evaluate whether humans have higher AMTL
    # Negative, significant coef => non-human has *lower* log-odds of AMTL than humans
    # Positive, significant coef => non-human has *higher* AMTL than humans
    alpha = 0.05
    strong_evidence_higher = []
    strong_evidence_not_higher = []
    ambiguous = []

    for genus in non_human_genera:
        info = evidence[genus]
        coef = info["coef"]
        pval = info["pval"]
        if np.isnan(coef) or np.isnan(pval):
            ambiguous.append(genus)
            continue
        if pval < alpha:
            if coef < 0:
                strong_evidence_higher.append(genus)
            else:
                strong_evidence_not_higher.append(genus)
        else:
            ambiguous.append(genus)

    # Decide Likert response
    if strong_evidence_higher and not strong_evidence_not_higher:
        # Consistent, significant evidence humans have higher AMTL than at least one non-human genus
        response = 80
        answer_text = (
            "Yes – the binomial regression provides strong evidence that modern "
            "humans have higher antemortem tooth loss frequencies than at least "
            "one non-human primate genus when controlling for age, sex, and tooth class."
        )
    elif strong_evidence_not_higher and not strong_evidence_higher:
        # Significant evidence against humans having higher AMTL
        response = 10
        answer_text = (
            "No – the binomial regression provides strong evidence that modern "
            "humans do not have higher antemortem tooth loss frequencies than "
            "non-human primates, and in some comparisons non-human genera show "
            "equal or higher AMTL when controlling for age, sex, and tooth class."
        )
    elif strong_evidence_higher and strong_evidence_not_higher:
        # Mixed evidence across genera
        response = 50
        answer_text = (
            "Unclear – the binomial regression yields mixed evidence: for some "
            "non-human genera humans appear to have higher antemortem tooth loss "
            "frequencies, while for others the opposite is true, after controlling "
            "for age, sex, and tooth class."
        )
    else:
        # No strong evidence either way
        response = 30
        answer_text = (
            "Probably not – the binomial regression does not show consistent, "
            "statistically significant differences indicating that modern humans "
            "have higher antemortem tooth loss frequencies than non-human primates "
            "once age, sex, and tooth class are accounted for."
        )

    # Build detailed explanation with key numerical summaries
    human_prob = genus_pred.get("Homo sapiens", float("nan"))
    lines = []
    lines.append(answer_text)
    lines.append(
        f"Model: a binomial generalized linear model of the proportion of missing teeth "
        f"(num_amtl / sockets) with sockets as the binomial denominator, including predictors "
        f"for genus (Homo sapiens as reference), age at death, estimated sex (prob_male), "
        f"and tooth class. The model was fit using all rows with sockets > 0."
    )
    lines.append(
        f"Predicted AMTL probabilities per tooth socket at mean age and sex and the most common "
        f"tooth class ({ref_tooth_class}) were approximately: "
        + ", ".join(
            f"{genus}: {prob:.3f}"
            for genus, prob in genus_pred.items()
        )
        + "."
    )
    lines.append(
        "Genus effects are interpreted relative to humans: negative coefficients mean the "
        "non-human genus has lower AMTL than humans, and positive coefficients mean higher AMTL."
    )
    for genus in non_human_genera:
        info = evidence[genus]
        coef = info["coef"]
        pval = info["pval"]
        lines.append(
            f"For {genus}, the genus coefficient relative to humans was "
            f"{coef:.3f} (p = {pval:.3g}), indicating "
            + (
                "significantly lower AMTL than humans."
                if (not np.isnan(coef) and not np.isnan(pval) and pval < alpha and coef < 0)
                else "significantly higher AMTL than humans."
                if (not np.isnan(coef) and not np.isnan(pval) and pval < alpha and coef > 0)
                else "no statistically significant difference from humans."
            )
        )
    if not np.isnan(human_prob):
        lines.append(
            f"Overall, the modeled human AMTL probability per socket was about {human_prob:.3f}, "
            f"which should be compared to the corresponding values for Pan, Pongo, and Papio "
            f"when judging whether humans truly have higher AMTL frequencies."
        )

    explanation = " ".join(lines)
    return response, explanation


def main():
    df = load_data(Path("amtl.csv"))
    result = fit_binomial_model(df)
    genus_params, genus_pvalues, genus_pred, ref_tooth_class = summarize_genus_effects(result, df)
    response, explanation = build_conclusion(
        genus_params=genus_params,
        genus_pvalues=genus_pvalues,
        genus_pred=genus_pred,
        ref_tooth_class=ref_tooth_class,
    )

    conclusion = {"response": int(response), "explanation": explanation}
    with open("conclusion.txt", "w", encoding="utf-8") as f:
        json.dump(conclusion, f)


if __name__ == "__main__":
    main()
