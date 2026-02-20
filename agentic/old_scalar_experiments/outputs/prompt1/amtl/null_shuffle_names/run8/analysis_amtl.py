import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def fit_amtl_model(csv_path: str):
    df = pd.read_csv(csv_path)

    n_raw = len(df)

    # Interpret columns based on info.json metadata.
    # - genus: number of missing teeth of given class
    # - age: number of observable sockets
    # - pop: estimated age at death
    # - stdev_age: estimated probability specimen is male
    # - sockets: tooth class (Anterior/Posterior/Premolar)
    # - tooth_class: taxonomic genus (Homo sapiens, Pan, Papio, Pongo)
    df["num_missing"] = df["genus"].astype(float)
    df["total_sockets"] = df["age"].astype(float)
    df["age_at_death"] = df["pop"].astype(float)
    df["prob_male"] = df["stdev_age"].astype(float)

    # Drop clearly inconsistent rows where missing teeth exceed observable sockets.
    valid_mask = df["num_missing"] <= df["total_sockets"]
    n_valid = int(valid_mask.sum())
    df = df[valid_mask].copy()

    # Build design matrix for binomial regression.
    genus_dummies = pd.get_dummies(df["tooth_class"], prefix="genus", drop_first=True)
    tooth_dummies = pd.get_dummies(df["sockets"], prefix="tooth", drop_first=True)

    X = pd.concat(
        [
            pd.DataFrame(
                {
                    "age_at_death": df["age_at_death"],
                    "prob_male": df["prob_male"],
                }
            ),
            genus_dummies,
            tooth_dummies,
        ],
        axis=1,
    )

    X = sm.add_constant(X, has_constant="add")

    # Response as [successes, failures] for binomial GLM.
    successes = df["num_missing"].astype(float)
    failures = (df["total_sockets"] - df["num_missing"]).astype(float)
    y = np.column_stack([successes, failures])

    model = sm.GLM(y, X, family=sm.families.Binomial())
    result = model.fit()

    return {
        "model_result": result,
        "X": X,
        "n_raw": int(n_raw),
        "n_valid": n_valid,
    }


def summarize_genus_effects(model_result, X):
    # Genus dummy coefficients: differences in log-odds relative to baseline genus.
    genus_cols = [c for c in X.columns if c.startswith("genus_")]
    genus_effects = {}
    for col in genus_cols:
        genus_label = col.split("genus_", 1)[1]
        coef = float(model_result.params[col])
        pval = float(model_result.pvalues[col])
        genus_effects[genus_label] = {"coef": coef, "p_value": pval}
    return genus_effects


def predict_genus_probabilities(model_result, X, df):
    # Use median age, median sex estimate, and modal tooth class
    # to get comparable predicted AMTL frequencies across genera.
    median_age = float(df["age_at_death"].median())
    median_prob_male = float(df["prob_male"].median())
    mode_tooth_class = str(df["sockets"].mode().iloc[0])

    genus_cols = [c for c in X.columns if c.startswith("genus_")]
    tooth_cols = [c for c in X.columns if c.startswith("tooth_")]

    genera = ["Homo sapiens", "Pan", "Papio", "Pongo"]
    rows = []
    index_labels = []

    for g in genera:
        row = {"const": 1.0, "age_at_death": median_age, "prob_male": median_prob_male}

        # Genus indicators (baseline genus has all zeros).
        for col in genus_cols:
            genus_label = col.split("genus_", 1)[1]
            row[col] = 1.0 if g == genus_label else 0.0

        # Tooth class indicators (baseline tooth class has all zeros).
        for col in tooth_cols:
            tooth_label = col.split("tooth_", 1)[1]
            row[col] = 1.0 if mode_tooth_class == tooth_label else 0.0

        rows.append(row)
        index_labels.append(g)

    X_pred = pd.DataFrame(rows, index=index_labels, columns=X.columns)
    pred_probs = model_result.predict(X_pred)

    return {genus: float(prob) for genus, prob in zip(index_labels, pred_probs)}


def build_explanation(
    n_raw,
    n_valid,
    genus_effects,
    genus_probs,
    response,
):
    dropped = n_raw - n_valid
    lines = []
    lines.append(
        "I analyzed the AMTL dataset ({} rows, {} used after dropping {} rows where the recorded number of missing teeth exceeded observable sockets).".format(
            n_raw,
            n_valid,
            dropped,
        )
    )
    lines.append(
        "I modeled the number of missing teeth out of observable sockets using a binomial regression (logit link), "
        "including taxonomic genus (Homo sapiens, Pan, Papio, Pongo), estimated age at death, sex estimate, and tooth class (anterior/posterior/premolar) as predictors."
    )

    homo_prob = genus_probs["Homo sapiens"]
    lines.append(
        "Using the fitted model, at the median age and sex estimate and the most common tooth class, "
        "the estimated AMTL frequency for Homo sapiens is {:.3f} missing teeth per socket.".format(
            homo_prob
        )
    )

    for genus in ["Pan", "Papio", "Pongo"]:
        prob = genus_probs[genus]
        diff = homo_prob - prob
        effect = genus_effects.get(genus)
        if effect is not None:
            coef = effect["coef"]
            pval = effect["p_value"]
            lines.append(
                "Relative to Homo sapiens, {} has a lower modeled AMTL frequency (estimated {:.3f}; difference {:.3f}), "
                "with a genus coefficient of {:.3f} (p = {:.4f}) in the binomial regression (negative coefficients indicate lower AMTL than humans).".format(
                    genus,
                    prob,
                    diff,
                    coef,
                    pval,
                )
            )
        else:
            lines.append(
                "Relative to Homo sapiens, {} has a lower modeled AMTL frequency (estimated {:.3f}; difference {:.3f}).".format(
                    genus,
                    prob,
                    diff,
                )
            )

    if response == "Yes":
        lines.append(
            "Because all non-human genera show lower modeled AMTL frequencies than Homo sapiens and their genus coefficients are negative and statistically significant, "
            "the data support the conclusion that modern humans have higher AMTL frequencies than Pan, Papio, and Pongo after accounting for age, sex, and tooth class."
        )
    else:
        lines.append(
            "Because at least one non-human genus does not show a statistically significant lower modeled AMTL frequency than Homo sapiens, "
            "the data do not provide clear evidence that humans have uniformly higher AMTL frequencies than all three non-human genera after accounting for age, sex, and tooth class."
        )

    return " ".join(lines)


def main():
    csv_path = "amtl.csv"
    analysis = fit_amtl_model(csv_path)
    result = analysis["model_result"]
    X = analysis["X"]
    n_raw = analysis["n_raw"]
    n_valid = analysis["n_valid"]

    # Reconstruct the filtered dataframe used inside fit_amtl_model
    df = pd.read_csv(csv_path)
    df["num_missing"] = df["genus"].astype(float)
    df["total_sockets"] = df["age"].astype(float)
    df["age_at_death"] = df["pop"].astype(float)
    df["prob_male"] = df["stdev_age"].astype(float)
    df = df[df["num_missing"] <= df["total_sockets"]].copy()

    genus_effects = summarize_genus_effects(result, X)
    genus_probs = predict_genus_probabilities(result, X, df)

    # Determine if humans have higher AMTL than each non-human genus.
    significant_all = True
    for genus in ["Pan", "Papio", "Pongo"]:
        effect = genus_effects.get(genus)
        if effect is None:
            significant_all = False
            break
        coef = effect["coef"]
        pval = effect["p_value"]
        # Non-human genus must have significantly lower log-odds of AMTL than Homo sapiens.
        if not (coef < 0 and pval < 0.05):
            significant_all = False
            break

    response = "Yes" if significant_all else "No"

    explanation = build_explanation(
        n_raw=n_raw,
        n_valid=n_valid,
        genus_effects=genus_effects,
        genus_probs=genus_probs,
        response=response,
    )

    conclusion = {"response": response, "explanation": explanation}

    conclusion_path = Path("conclusion.txt")
    with conclusion_path.open("w") as f:
        json.dump(conclusion, f)


if __name__ == "__main__":
    main()

