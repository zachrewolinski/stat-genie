import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def load_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    # Basic cleaning: drop clearly invalid rows
    df = df.copy()
    df = df[df["sockets"] > 0]
    df = df[df["num_amtl"] >= 0]
    df = df[df["num_amtl"] <= df["sockets"]]

    return df


def fit_binomial_glm(df: pd.DataFrame):
    df = df.copy()

    # Proportion of missing teeth out of observable sockets
    df["amtl_prop"] = df["num_amtl"] / df["sockets"]

    # Ensure categorical encodings and set Homo sapiens as reference level
    df["genus"] = df["genus"].astype("category")
    if "Homo sapiens" in list(df["genus"].cat.categories):
        # Put Homo sapiens first so it becomes the baseline
        other = [g for g in df["genus"].cat.categories if g != "Homo sapiens"]
        df["genus"] = df["genus"].cat.set_categories(
            ["Homo sapiens", *other], ordered=False
        )

    df["tooth_class"] = df["tooth_class"].astype("category")

    formula = "amtl_prop ~ C(genus) + age + prob_male + C(tooth_class)"
    model = smf.glm(
        formula=formula,
        data=df,
        family=sm.families.Binomial(),
        freq_weights=df["sockets"],
    )

    # Fit with cluster-robust covariance at the specimen level to account for
    # multiple tooth-class observations per individual.
    robust_res = model.fit(
        cov_type="cluster", cov_kwds={"groups": df["specimen"]}
    )
    return robust_res, df


def estimate_genus_effects(robust_res, df: pd.DataFrame):
    params = robust_res.params
    bse = robust_res.bse

    # With Homo sapiens as baseline, coefficients for other genera indicate
    # log-odds differences relative to humans.
    genus_levels = list(df["genus"].cat.categories)

    # Build a simple reference profile for prediction
    ref_age = float(df["age"].median())
    ref_prob_male = float(df["prob_male"].mean())
    common_tooth_class = df["tooth_class"].mode()[0]

    prediction_rows = []
    for g in genus_levels:
        prediction_rows.append(
            {
                "genus": g,
                "age": ref_age,
                "prob_male": ref_prob_male,
                "tooth_class": common_tooth_class,
                "amtl_prop": 0.0,  # placeholder, not used in prediction
                "sockets": 1.0,
            }
        )

    pred_df = pd.DataFrame(prediction_rows)
    pred_probs = robust_res.predict(pred_df)

    genus_pred = {
        g: float(p) for g, p in zip(genus_levels, pred_probs)
    }

    # Collect coefficient information for non-human genera
    non_human_info = {}
    for g in genus_levels:
        if g == "Homo sapiens":
            continue
        term = f"C(genus)[T.{g}]"
        if term in params.index:
            coef = float(params[term])
            se = float(bse[term])
            z = coef / se if se > 0 else np.nan
            non_human_info[g] = {
                "coef_vs_human": coef,
                "se": se,
                "z": z,
            }

    return genus_pred, non_human_info


def decide_answer(genus_pred, non_human_info):
    human_rate = genus_pred.get("Homo sapiens", np.nan)

    # Determine whether humans clearly have higher AMTL frequency than all
    # non-human genera after adjustment.
    higher_than_all = True
    strong_evidence = True

    for genus, pred in genus_pred.items():
        if genus == "Homo sapiens":
            continue

        # If any non-human genus has an equal or higher predicted
        # AMTL proportion, humans are not clearly higher.
        if pred >= human_rate:
            higher_than_all = False

        info = non_human_info.get(genus)
        if info is not None:
            # Coefficient > 0 means that genus has higher AMTL than humans.
            # Check for strong evidence in either direction using |z| > 2.
            if abs(info["z"]) <= 2:
                strong_evidence = False

    if higher_than_all and strong_evidence:
        response = "Yes"
        confidence = 80
        explanation = (
            "A binomial regression of AMTL proportion on genus, age, "
            "sex proxy, and tooth class indicates that Homo sapiens has "
            "consistently higher predicted AMTL frequencies than each "
            "non-human primate genus, with genus effects that are both "
            "directionally and statistically strong."
        )
    elif not higher_than_all and strong_evidence:
        response = "No"
        confidence = 85
        explanation = (
            "A binomial regression of AMTL proportion on genus, age, "
            "sex proxy, and tooth class shows that at least one "
            "non-human primate genus has AMTL frequencies that are "
            "comparable to or higher than Homo sapiens, and the genus "
            "coefficients provide strong evidence against humans having "
            "uniformly higher AMTL after adjustment."
        )
    else:
        response = "No"
        confidence = 70
        explanation = (
            "A binomial regression of AMTL proportion on genus, age, "
            "sex proxy, and tooth class does not provide strong, "
            "consistent evidence that Homo sapiens has higher AMTL "
            "frequencies than all non-human primate genera after "
            "adjustment; estimated differences are small or "
            "statistically uncertain for at least one comparison."
        )

    # Attach a brief quantitative summary for transparency
    explanation += (
        " Predicted AMTL proportions for a reference profile are: "
        + ", ".join(f"{g}: {p:.3f}" for g, p in genus_pred.items())
        + ". "
    )

    if non_human_info:
        coef_parts = []
        for g, info in non_human_info.items():
            coef_parts.append(
                f"{g}: coef={info['coef_vs_human']:.3f}, z={info['z']:.2f}"
            )
        explanation += (
            "Non-human genus coefficients (log-odds vs humans): "
            + ", ".join(coef_parts)
            + "."
        )

    return response, confidence, explanation


def main():
    df = load_data(Path("amtl.csv"))
    robust_res, df_model = fit_binomial_glm(df)
    genus_pred, non_human_info = estimate_genus_effects(robust_res, df_model)
    response, confidence, explanation = decide_answer(genus_pred, non_human_info)

    conclusion = {
        "response": response,
        "confidence": confidence,
        "explanation": explanation,
    }

    Path("conclusion.txt").write_text(json.dumps(conclusion))


if __name__ == "__main__":
    main()
