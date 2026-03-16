import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main() -> None:
    data_path = Path("amtl.csv")
    raw_df = pd.read_csv(data_path)

    # Exclude clearly invalid rows where the number of missing teeth exceeds
    # the number of observable sockets, which violates the binomial model.
    invalid_mask = raw_df["num_amtl"] > raw_df["sockets"]
    n_invalid = int(invalid_mask.sum())
    df = raw_df.loc[~invalid_mask].copy()

    # Binary indicator for modern humans vs. non-human primates
    df["is_human"] = (df["genus"] == "Homo sapiens").astype(int)
    df["prop_amtl"] = df["num_amtl"] / df["sockets"]

    # Expand counts so that each observable socket is treated as a Bernoulli trial
    expanded_rows = []
    for _, row in df.iterrows():
        sockets = int(row["sockets"])
        missing = int(row["num_amtl"])
        if sockets <= 0:
            continue
        if missing < 0:
            missing = 0
        if missing > sockets:
            missing = sockets
        for i in range(sockets):
            expanded_rows.append(
                {
                    "is_missing": 1 if i < missing else 0,
                    "age": row["age"],
                    "prob_male": row["prob_male"],
                    "tooth_class": row["tooth_class"],
                    "genus": row["genus"],
                    "specimen": row["specimen"],
                    "is_human": row["is_human"],
                }
            )

    df_long = pd.DataFrame(expanded_rows)

    # Binomial regression: probability that a socket is missing
    formula = "is_missing ~ is_human + age + prob_male + C(tooth_class)"
    model = smf.glm(
        formula=formula,
        data=df_long,
        family=sm.families.Binomial(),
    )
    result = model.fit()

    # Extract key statistics for the human vs non-human effect
    coef_human = float(result.params["is_human"])
    pval_human = float(result.pvalues["is_human"])

    # Predicted AMTL probabilities for humans vs non-humans at average covariate values,
    # averaging over the empirical distribution of tooth classes.
    mean_age = float(df["age"].mean())
    mean_prob_male = float(df["prob_male"].mean())
    tooth_weights = df["tooth_class"].value_counts(normalize=True)

    rows = []
    for is_human_val in (0, 1):
        for tooth_class, weight in tooth_weights.items():
            rows.append(
                {
                    "is_human": is_human_val,
                    "age": mean_age,
                    "prob_male": mean_prob_male,
                    "tooth_class": tooth_class,
                    "weight": float(weight),
                }
            )

    pred_df = pd.DataFrame(rows)
    pred_df["pred"] = result.predict(pred_df)

    pred_nonhuman = float(
        (pred_df[pred_df["is_human"] == 0]["pred"] * pred_df[pred_df["is_human"] == 0]["weight"]).sum()
    )
    pred_human = float(
        (pred_df[pred_df["is_human"] == 1]["pred"] * pred_df[pred_df["is_human"] == 1]["weight"]).sum()
    )
    diff_pred = pred_human - pred_nonhuman

    # Basic sample description (after excluding invalid rows)
    n_rows = int(len(df))
    n_specimens = int(df["specimen"].nunique())
    genus_counts = df["genus"].value_counts()
    genus_summary = ", ".join(f"{g}: {int(c)}" for g, c in genus_counts.items())
    total_sockets = int(df["sockets"].sum())
    total_missing = int(df["num_amtl"].sum())

    # Map statistical evidence to Likert-scale response (0 = strong No, 100 = strong Yes).
    # Direction is determined by whether humans have higher (diff_pred > 0) or lower AMTL.
    if pval_human >= 0.05:
        # No statistically significant difference after controlling for covariates.
        if abs(diff_pred) < 0.01:
            response_value = 20  # very little evidence for higher human AMTL
        elif abs(diff_pred) < 0.03:
            response_value = 30
        else:
            response_value = 40
    else:
        # Statistically significant difference; strength scales with effect size.
        abs_diff = abs(diff_pred)
        if abs_diff < 0.02:
            base = 65
        elif abs_diff < 0.05:
            base = 75
        else:
            base = 88

        if diff_pred > 0:
            # Humans have higher predicted AMTL
            response_value = base
        else:
            # Humans have lower predicted AMTL
            response_value = 100 - base

    response_value = int(round(response_value))

    # Construct textual explanation
    direction_text = (
        "higher" if diff_pred > 0 else "lower" if diff_pred < 0 else "similar"
    )
    significance_text = (
        "not statistically significant (p ≥ 0.05)"
        if pval_human >= 0.05
        else "statistically significant (p < 0.05)"
    )

    explanation = (
        "Research question: Do modern humans (Homo sapiens) have higher frequencies of "
        "antemortem tooth loss (AMTL) than non-human primates (Pan, Pongo, Papio) after "
        "accounting for age, sex, and tooth class?\n\n"
        f"Data and model: I analyzed the provided AMTL dataset with {n_rows} tooth-class records "
        f"from {n_specimens} unique specimens (genus counts: {genus_summary}) after excluding "
        f"{n_invalid} records where the recorded number of missing teeth exceeded the number of "
        "observable sockets, which is incompatible with a binomial model. Across the retained records, "
        f"there are {total_sockets} observable tooth sockets and {total_missing} missing teeth. "
        "For modeling, I expanded each record so that every observable socket is treated as a single "
        "Bernoulli trial (missing vs. present) and fit a binomial logistic regression with logit link. "
        "The predictors were a binary indicator "
        "for modern humans vs. non-human primates (is_human), age at death, estimated sex "
        "(probability of being male), and categorical tooth class (Anterior, Posterior, Premolar). "
        "The model was fit using the number of sockets as binomial trials.\n\n"
        f"Key result: The coefficient for the human indicator (is_human) is {coef_human:.3f} "
        f"with p-value {pval_human:.3g}, which is {significance_text}. Using the fitted model, "
        f"the predicted AMTL frequency at average age and sex, averaged over the observed distribution "
        f"of tooth classes, is {pred_human:.3f} for modern humans and {pred_nonhuman:.3f} for "
        f"non-human primates, a difference of {diff_pred:.3f} (humans {direction_text}).\n\n"
        "Interpretation: "
    )

    if pval_human >= 0.05:
        explanation += (
            "Because the human vs. non-human coefficient is not statistically significant and the "
            "difference in predicted AMTL frequencies is small, the data do not provide strong evidence "
            "that modern humans have higher AMTL frequencies than non-human primates once age, sex, and "
            "tooth class are accounted for. Any observed differences could plausibly be due to sampling "
            "variation rather than a true genus-level effect."
        )
    elif diff_pred > 0:
        explanation += (
            "Because the human indicator is positive and statistically significant, and the predicted "
            "AMTL frequency for humans is higher than for non-human primates, the analysis supports the "
            "conclusion that modern humans do have higher AMTL frequencies after adjusting for age, sex, "
            "and tooth class. The magnitude of the difference on the probability scale is modest but "
            "consistent with a real genus-level effect."
        )
    else:
        explanation += (
            "Because the human indicator is negative and statistically significant, and the predicted "
            "AMTL frequency for humans is lower than for non-human primates, the analysis supports the "
            "conclusion that modern humans do not have higher AMTL frequencies after adjusting for age, "
            "sex, and tooth class; if anything, humans show lower AMTL frequencies in this sample."
        )

    conclusion = {
        "response": response_value,
        "explanation": explanation,
    }

    conclusion_path = Path("conclusion.txt")
    conclusion_path.write_text(json.dumps(conclusion, ensure_ascii=False))


if __name__ == "__main__":
    main()
