import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    # Basic cleaning: ensure valid counts and proportions
    df = df.copy()
    df = df[df["sockets"] > 0]
    df["prop_amtl"] = df["num_amtl"] / df["sockets"]

    # Clip tiny numerical issues outside [0, 1]
    df["prop_amtl"] = df["prop_amtl"].clip(0.0, 1.0)

    df["genus"] = df["genus"].astype("category")
    df["tooth_class"] = df["tooth_class"].astype("category")
    return df


def fit_model(df: pd.DataFrame):
    # Binomial regression on proportions with sockets as binomial denominator
    formula = (
        "prop_amtl ~ C(genus, Treatment(reference='Homo sapiens'))"
        " + C(tooth_class) + age + prob_male"
    )
    model = smf.glm(
        formula=formula,
        data=df,
        family=sm.families.Binomial(),
        freq_weights=df["sockets"],
    )
    result = model.fit()
    return result


def compute_genus_predictions(df: pd.DataFrame, result) -> pd.Series:
    # Average predicted AMTL probability per genus, accounting for covariates
    df = df.copy()
    df["pred_prob"] = result.predict(df)
    mean_probs = df.groupby("genus")["pred_prob"].mean()
    return mean_probs


def summarize_results(mean_probs: pd.Series) -> dict:
    human_label = "Homo sapiens"
    if human_label not in mean_probs.index:
        # If for some reason there are no humans, we cannot answer as posed.
        response = "No"
        confidence = 10
        explanation = (
            "The dataset does not contain any Homo sapiens specimens, so it is "
            "not possible to compare AMTL frequencies between humans and "
            "non-human primates."
        )
        return {
            "response": response,
            "confidence": confidence,
            "explanation": explanation,
        }

    human_prob = float(mean_probs.loc[human_label])
    nonhuman = {g: float(p) for g, p in mean_probs.items() if g != human_label}

    if not nonhuman:
        response = "No"
        confidence = 10
        explanation = (
            "The dataset does not contain any non-human primate genera, so it "
            "is not possible to compare AMTL frequencies."
        )
        return {
            "response": response,
            "confidence": confidence,
            "explanation": explanation,
        }

    max_nonhuman_label = max(nonhuman, key=nonhuman.get)
    max_nonhuman_prob = nonhuman[max_nonhuman_label]

    higher_than_all = human_prob > max_nonhuman_prob
    delta = abs(human_prob - max_nonhuman_prob)

    # Heuristic confidence based on effect size (difference in predicted probabilities)
    # and presence of all four genera.
    has_all_genera = {"Pan", "Papio", "Pongo"}.issubset(set(nonhuman.keys()))

    # Map difference in probabilities into a 0–100 scale and
    # keep it in a reasonable range.
    base_conf = min(99, max(50, int(50 + (delta / 0.1) * 20)))
    if has_all_genera:
        base_conf = min(99, base_conf + 10)

    response = "Yes" if higher_than_all else "No"

    explanation_lines = [
        "Fitted a binomial regression model for the proportion of missing teeth "
        "(num_amtl / sockets) using genus, tooth class, age, and estimated sex "
        "(prob_male) as predictors.",
        "From the model, I computed adjusted mean predicted AMTL probabilities "
        "for each genus by averaging predicted probabilities across specimens.",
        f"The adjusted mean AMTL probability for Homo sapiens was "
        f"{human_prob:.3f}.",
    ]

    for genus, prob in sorted(nonhuman.items()):
        explanation_lines.append(
            f"The adjusted mean AMTL probability for {genus} was {prob:.3f}."
        )

    if higher_than_all:
        explanation_lines.append(
            "Homo sapiens showed a higher adjusted AMTL probability than each "
            "non-human primate genus (Pan, Pongo, Papio) in this dataset, even "
            "after accounting for age, sex, and tooth class in the regression "
            "model."
        )
    else:
        explanation_lines.append(
            "Homo sapiens did not show a higher adjusted AMTL probability than "
            "all non-human primate genera after accounting for age, sex, and "
            "tooth class in the regression model."
        )

    explanation = " ".join(explanation_lines)

    return {
        "response": response,
        "confidence": int(base_conf),
        "explanation": explanation,
    }


def write_conclusion(conclusion: dict, path: str = "conclusion.txt") -> None:
    # Write ONLY the JSON object to the file, with no extra lines.
    with open(path, "w", encoding="utf-8") as f:
        json.dump(conclusion, f, ensure_ascii=False)


def main():
    csv_path = Path("amtl.csv")
    if not csv_path.exists():
        raise FileNotFoundError("amtl.csv not found in the current directory.")

    df = load_data(str(csv_path))
    result = fit_model(df)

    # Print a short summary to stdout for transparency/debugging.
    print(result.summary())

    mean_probs = compute_genus_predictions(df, result)
    conclusion = summarize_results(mean_probs)
    write_conclusion(conclusion)


if __name__ == "__main__":
    main()

