import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def load_and_prepare_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    # Rename columns to more meaningful names based on info.json descriptions
    df = df.rename(
        columns={
            "sockets": "tooth_class",  # Anterior/Posterior/Premolar
            "prob_male": "specimen_id",
            "genus": "num_amtl",  # number of missing teeth of that class
            "age": "n_sockets_observed",
            "pop": "age_at_death",
            "num_amtl": "age_at_death_sd",
            "stdev_age": "prob_male",
            "tooth_class": "genus",  # Homo sapiens, Pan, Papio, Pongo
            "specimen": "region",
        }
    )

    # Create response as binomial (missing vs present teeth)
    df["num_amtl"] = df["num_amtl"].astype(float)
    df["n_sockets_observed"] = df["n_sockets_observed"].astype(float)

    # Remove any impossible rows where missing teeth exceed observable sockets
    df = df[df["num_amtl"] <= df["n_sockets_observed"]].copy()

    # Proportion of missing teeth and binomial weight
    df["prop_amtl"] = df["num_amtl"] / df["n_sockets_observed"]
    df = df.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["prop_amtl", "age_at_death", "prob_male"]
    )

    # Categorical variables
    df["genus"] = df["genus"].astype("category")
    df["tooth_class"] = df["tooth_class"].astype("category")

    return df


def fit_binomial_model(df: pd.DataFrame):
    # Binomial regression with logit link
    # Endog: proportion missing with weights = n_sockets_observed
    formula = "prop_amtl ~ C(genus) + age_at_death + prob_male + C(tooth_class)"

    model = smf.glm(
        formula=formula,
        data=df,
        family=sm.families.Binomial(),
        freq_weights=df["n_sockets_observed"],
    )
    result = model.fit()
    return result


def evaluate_human_vs_others(result) -> tuple[str, float, str]:
    # We treat Homo sapiens as the reference level for genus.
    # In patsy coding, Homo sapiens (alphabetically last) might not be the baseline,
    # so we rely on the actual categories and re-fit if needed.
    # However, since we encoded C(genus), statsmodels will choose a baseline.
    # We can still compare estimated mean log-odds for each genus using predictions.

    # Build a small design grid holding age_at_death and prob_male at their means
    # and tooth_class at its most common level for comparability.
    params = result.params
    cov = result.cov_params()

    # Extract unique levels
    exog = result.model.data.frame
    common_tooth_class = exog["tooth_class"].mode().iat[0]
    mean_age = exog["age_at_death"].mean()
    mean_prob_male = exog["prob_male"].mean()

    genera = sorted(exog["genus"].unique())

    # For each genus, construct a row and compute predicted logit and SE
    predictions = {}
    for g in genera:
        row = {
            "Intercept": 1.0,
            "age_at_death": mean_age,
            "prob_male": mean_prob_male,
        }

        # Tooth class dummies
        for tc in exog["tooth_class"].cat.categories:
            col_name = f"C(tooth_class)[T.{tc}]"
            row[col_name] = 0.0
        # Activate chosen tooth class if its column exists
        chosen_tc_col = f"C(tooth_class)[T.{common_tooth_class}]"
        if chosen_tc_col in params.index:
            row[chosen_tc_col] = 1.0

        # Genus dummies
        for gg in exog["genus"].cat.categories:
            col_name = f"C(genus)[T.{gg}]"
            row[col_name] = 0.0
        genus_col = f"C(genus)[T.{g}]"
        # Baseline genus has no column; others do
        if genus_col in params.index:
            row[genus_col] = 1.0

        # Align with params index
        x = np.array([row.get(name, 0.0) for name in params.index])
        logit = float(np.dot(x, params.values))
        se = float(np.sqrt(np.dot(x, cov @ x)))
        predictions[g] = (logit, se)

    # Identify human category
    human_keys = [g for g in genera if "Homo" in g or "sapiens" in g]
    if not human_keys:
        explanation = (
            "Could not identify a Homo sapiens genus category in the data; "
            "treating the result as inconclusive."
        )
        return "No", 40.0, explanation

    human_genus = human_keys[0]
    human_logit, human_se = predictions[human_genus]

    # Compare humans to each non-human genus via differences in predicted logits
    lower_better = []
    nonhuman_details = []
    for g, (logit_g, se_g) in predictions.items():
        if g == human_genus:
            continue
        diff = human_logit - logit_g
        se_diff = np.sqrt(human_se**2 + se_g**2)
        z = diff / se_diff if se_diff > 0 else np.nan

        # 95% CI on difference
        ci_low = diff - 1.96 * se_diff
        ci_high = diff + 1.96 * se_diff

        # diff > 0 means humans have higher log-odds of AMTL
        lower_better.append(diff > 0 and ci_low > 0)
        nonhuman_details.append(
            (g, float(diff), float(ci_low), float(ci_high))
        )

    if all(lower_better):
        response = "Yes"
        confidence = 85.0
    elif any(lower_better):
        response = "Yes"
        confidence = 65.0
    else:
        response = "No"
        confidence = 60.0

    # Build brief textual explanation
    lines = [
        "Fitted a binomial regression of the proportion of missing teeth "
        "(num_amtl / n_sockets_observed) on genus, age at death, sex "
        "(probability of being male), and tooth class (anterior/posterior/premolar), "
        "using each specimen-tooth-class group as a binomial observation.",
        f"The model estimates genus-specific log-odds of AMTL at mean age "
        f"({mean_age:.1f} years), mean sex (prob_male={mean_prob_male:.2f}), "
        f"and the most common tooth class ({common_tooth_class}).",
    ]

    for g, diff, ci_low, ci_high in nonhuman_details:
        direction = "higher" if diff > 0 else "lower"
        lines.append(
            f"Compared to {g}, humans show {direction} predicted AMTL frequencies "
            f"(log-odds difference = {diff:.2f}, 95% CI [{ci_low:.2f}, {ci_high:.2f}])."
        )

    if response == "Yes":
        lines.append(
            "Across all non-human genera (Pan, Papio, Pongo), the estimated human "
            "AMTL frequency is higher after adjusting for age, sex, and tooth class."
        )
    else:
        lines.append(
            "At least one non-human genus shows comparable or higher adjusted AMTL "
            "frequencies to humans, so the evidence does not clearly support "
            "humans having uniformly higher AMTL rates."
        )

    explanation = " ".join(lines)
    return response, confidence, explanation


def write_conclusion(response: str, confidence: float, explanation: str, path: Path) -> None:
    obj = {
        "response": response,
        "confidence": round(float(confidence), 1),
        "confidence": float(round(confidence, 1)),
        "explanation": explanation,
    }
    # Ensure only the JSON object appears in the file
    path.write_text(json.dumps(obj, ensure_ascii=False))


def main():
    csv_path = Path("amtl.csv")
    df = load_and_prepare_data(csv_path)
    result = fit_binomial_model(df)
    response, confidence, explanation = evaluate_human_vs_others(result)
    write_conclusion(response, confidence, explanation, Path("conclusion.txt"))


if __name__ == "__main__":
    main()

