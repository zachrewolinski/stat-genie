import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def compute_likert_score(coef: float, pval: float, diff_mean: float) -> int:
    """
    Map direction, significance, and average probability difference
    to a 0–100 Likert score (0=strong No, 100=strong Yes).
    """
    score: int

    if coef > 0:
        # Evidence humans have higher AMTL
        if pval < 1e-6 and diff_mean > 0.05:
            score = 95
        elif pval < 1e-3 and diff_mean > 0.03:
            score = 85
        elif pval < 1e-2 and diff_mean > 0.02:
            score = 75
        elif pval < 0.05 and diff_mean > 0.01:
            score = 65
        else:
            score = 55
    elif coef < 0:
        # Evidence humans have lower AMTL
        if pval < 1e-6 and diff_mean < -0.05:
            score = 5
        elif pval < 1e-3 and diff_mean < -0.03:
            score = 15
        elif pval < 1e-2 and diff_mean < -0.02:
            score = 25
        elif pval < 0.05 and diff_mean < -0.01:
            score = 35
        else:
            score = 45
    else:
        score = 50

    return int(max(0, min(100, score)))


def main() -> None:
    data_path = Path("amtl.csv")
    if not data_path.exists():
        raise FileNotFoundError("amtl.csv not found in current directory.")

    df = pd.read_csv(data_path)

    # Relabel columns to their semantic meanings based on info.json description.
    df = df.copy()
    df["tooth_class_cat"] = df["sockets"]  # Anterior / Posterior / Premolar
    df["specimen_id"] = df["prob_male"]  # unique specimen identifier
    df["num_missing"] = df["genus"]  # number of missing teeth in this class
    df["n_sockets"] = df["age"]  # observable sockets for this tooth class
    df["age_at_death"] = df["pop"]  # estimated age at death
    df["age_uncertainty"] = df["num_amtl"]  # uncertainty of age estimate
    df["sex_est"] = df["stdev_age"]  # estimate of sex (treated as 0–1 scale)
    df["genus_label"] = df["tooth_class"]  # Homo sapiens / Pan / Papio / Pongo
    df["region"] = df["specimen"]  # region/population label

    # Focus on comparison between modern humans and the three non-human genera.
    target_genera = ["Homo sapiens", "Pan", "Papio", "Pongo"]
    df = df[df["genus_label"].isin(target_genera)].copy()

    # Valid binomial observations: positive number of sockets, and missing teeth
    # between 0 and n_sockets.
    df = df[(df["n_sockets"] > 0) & (df["num_missing"] >= 0)]
    df = df[df["num_missing"] <= df["n_sockets"]].copy()

    # Drop rows with missing key covariates.
    df = df.dropna(
        subset=["sex_est", "age_at_death", "tooth_class_cat", "num_missing", "n_sockets"]
    ).copy()

    # Indicator for modern humans vs non-human primates.
    df["is_human"] = (df["genus_label"] == "Homo sapiens").astype(int)

    # Compute overall descriptive AMTL frequencies by genus.
    genus_stats = (
        df.groupby("genus_label")
        .agg(
            total_missing=("num_missing", "sum"),
            total_sockets=("n_sockets", "sum"),
        )
        .assign(amtl_rate=lambda g: g["total_missing"] / g["total_sockets"])
    )

    # Design matrix: intercept, human indicator, age at death, sex estimate,
    # and tooth-class dummies (Anterior / Posterior / Premolar).
    X = pd.get_dummies(
        df[["is_human", "age_at_death", "sex_est", "tooth_class_cat"]],
        columns=["tooth_class_cat"],
        drop_first=True,
    )
    X = sm.add_constant(X, has_constant="add")

    # Binomial response as (successes, failures) = (missing teeth, present teeth).
    y = np.column_stack(
        [df["num_missing"].to_numpy(), (df["n_sockets"] - df["num_missing"]).to_numpy()]
    )

    model = sm.GLM(y, X, family=sm.families.Binomial())
    result = model.fit()

    coef = float(result.params["is_human"])
    se = float(result.bse["is_human"])
    pval = float(result.pvalues["is_human"])

    # Average difference in predicted AMTL probability for the same teeth
    # if they belonged to a human vs a non-human, holding covariates fixed.
    X_nonhuman = X.copy()
    X_nonhuman["is_human"] = 0
    pred_nonhuman = result.predict(X_nonhuman)

    X_human = X.copy()
    X_human["is_human"] = 1
    pred_human = result.predict(X_human)

    diff_mean = float((pred_human - pred_nonhuman).mean())

    score = compute_likert_score(coef=coef, pval=pval, diff_mean=diff_mean)

    # Prepare explanation string with key numerical evidence.
    human_rate = float(genus_stats.loc["Homo sapiens", "amtl_rate"])
    nonhuman_stats = genus_stats.loc[["Pan", "Papio", "Pongo"]]
    nonhuman_total_missing = int(nonhuman_stats["total_missing"].sum())
    nonhuman_total_sockets = int(nonhuman_stats["total_sockets"].sum())
    nonhuman_rate = float(nonhuman_total_missing / nonhuman_total_sockets)

    explanation = (
        "Research question: Do modern humans (Homo sapiens) have higher frequencies of "
        "antemortem tooth loss (AMTL) than non-human primates (Pan, Papio, Pongo) after "
        "accounting for age, sex, and tooth class? "
        "I modelled AMTL as a binomial outcome (number of missing teeth out of observable "
        "sockets in each record) with a generalized linear model using a binomial family "
        "and logit link. The predictors were an indicator for modern humans versus "
        "non-human genera, estimated age at death, a continuous sex estimate, and "
        "tooth-class indicators (Anterior/Posterior/Premolar).\n\n"
        f"Descriptively, the overall AMTL rate for Homo sapiens was "
        f"{human_rate:.3f}, while the combined rate for the non-human genera "
        f"(Pan, Papio, Pongo) was {nonhuman_rate:.3f} "
        f"({nonhuman_total_missing} missing teeth out of {nonhuman_total_sockets} "
        "sockets). In the regression model, the coefficient for the human indicator "
        f"was {coef:.3f} (standard error {se:.3f}, p-value {pval:.3g}), indicating "
        f"a {'higher' if coef > 0 else 'lower' if coef < 0 else 'similar'} AMTL "
        "log-odds for humans relative to non-human primates after adjusting for "
        "age, sex, and tooth class. On the probability scale, averaging over all "
        "observations, switching the genus from non-human to human while holding "
        f"covariates fixed changed the predicted AMTL probability by "
        f"{diff_mean:.3f} on average.\n\n"
        "I mapped this evidence (direction of the human effect, its statistical "
        "significance, and the average change in predicted AMTL probability) onto "
        "a 0–100 Likert scale, where 0 is a very strong 'No' and 100 is a very "
        "strong 'Yes'. The resulting score therefore reflects how strongly the "
        "data support the claim that modern humans have higher AMTL frequencies "
        "than the non-human primate genera, after accounting for age, sex, and "
        "tooth class."
    )

    conclusion = {"response": score, "explanation": explanation}

    with open("conclusion.txt", "w", encoding="utf-8") as f:
        json.dump(conclusion, f)


if __name__ == "__main__":
    main()

