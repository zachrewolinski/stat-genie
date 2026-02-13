import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def load_data(base_path: Path):
    info_path = base_path / "info.json"
    data_path = base_path / "amtl.csv"

    with info_path.open("r") as f:
        info = json.load(f)

    df = pd.read_csv(data_path)
    return info, df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Keep rows with valid counts and sockets
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(
        subset=["num_amtl", "sockets", "age", "prob_male", "genus", "tooth_class"]
    )
    df = df[df["sockets"] > 0]

    # Categorical variables
    df["tooth_class"] = df["tooth_class"].astype("category")
    df["genus"] = df["genus"].astype("category")

    # Ensure Homo sapiens is the reference (baseline) genus where present
    genus_order = ["Homo sapiens", "Pan", "Papio", "Pongo"]
    present = [g for g in genus_order if g in df["genus"].cat.categories]
    if present:
        df["genus"] = df["genus"].cat.set_categories(present, ordered=False)

    # Proportion of missing teeth for binomial model
    df["prop_amtl"] = df["num_amtl"] / df["sockets"]

    return df


def fit_model(df: pd.DataFrame):
    # Binomial GLM with sockets as frequency weights
    model = smf.glm(
        formula="prop_amtl ~ C(genus) + age + prob_male + C(tooth_class)",
        data=df,
        family=sm.families.Binomial(),
        freq_weights=df["sockets"],
    ).fit()
    return model


def summarize_genus_effects(model) -> dict:
    params = model.params
    conf_int = model.conf_int()
    pvalues = model.pvalues

    genus_effects = {}
    for genus in ["Pan", "Papio", "Pongo"]:
        term = f"C(genus)[T.{genus}]"
        if term in params.index:
            genus_effects[genus] = {
                "coef": float(params[term]),
                "ci_low": float(conf_int.loc[term, 0]),
                "ci_high": float(conf_int.loc[term, 1]),
                "pvalue": float(pvalues[term]),
            }

    return genus_effects


def compute_predicted_probs_by_genus(df: pd.DataFrame, model) -> dict:
    # Predicted AMTL proportion for each genus at mean age and prob_male
    mean_age = df["age"].mean()
    mean_prob_male = df["prob_male"].mean()

    # Use the most common tooth_class as representative
    common_class = df["tooth_class"].mode().iat[0]

    genera = df["genus"].cat.categories
    rows = []
    for g in genera:
        rows.append(
            {
                "genus": g,
                "age": mean_age,
                "prob_male": mean_prob_male,
                "tooth_class": common_class,
            }
        )

    pred_df = pd.DataFrame(rows)
    pred = model.predict(pred_df)

    return {g: float(p) for g, p in zip(pred_df["genus"], pred)}


def decide_conclusion(genus_effects: dict, pred_probs: dict) -> dict:
    # Determine if humans have higher AMTL frequency than each non-human genus.
    # In the GLM, Homo sapiens is the reference; negative coefficients for other genera
    # indicate lower AMTL compared to humans (i.e., humans higher).
    all_lower_and_significant = True
    any_positive = False
    any_positive_significant = False
    min_pvalue = 1.0

    for genus, stats in genus_effects.items():
        coef = stats["coef"]
        ci_low = stats["ci_low"]
        ci_high = stats["ci_high"]
        pval = stats["pvalue"]
        min_pvalue = min(min_pvalue, pval)

        if coef > 0:
            any_positive = True
        if coef > 0 and ci_low > 0:
            any_positive_significant = True
        if not (coef < 0 and ci_high < 0):
            all_lower_and_significant = False

    human_prob = pred_probs.get("Homo sapiens")
    nonhuman_probs = [
        pred_probs.get(g) for g in ["Pan", "Papio", "Pongo"] if g in pred_probs
    ]
    nonhuman_probs = [p for p in nonhuman_probs if p is not None]

    human_higher_pred = False
    human_lower_pred = False
    if human_prob is not None and nonhuman_probs:
        human_higher_pred = human_prob > max(nonhuman_probs)
        human_lower_pred = human_prob < max(nonhuman_probs)

    # Decide response: require humans to be higher than all non-human genera
    # (and with reasonably strong evidence) for a "Yes".
    if all_lower_and_significant and human_higher_pred:
        response = "Yes"
        # Strong statistical evidence and consistent predicted probabilities
        if min_pvalue < 1e-4:
            strength = 95
            confidence = 90
        elif min_pvalue < 0.01:
            strength = 85
            confidence = 80
        else:
            strength = 75
            confidence = 70
    else:
        # Any indication that non-human genera have equal or higher AMTL counts
        # supports a "No" answer to the question.
        response = "No"
        if human_lower_pred or any_positive:
            if any_positive_significant:
                strength = 85
                confidence = 80
            else:
                strength = 70
                confidence = 65
        else:
            # Effect directions are ambiguous; lean to a weaker "No".
            strength = 55
            confidence = 50

    return {
        "response": response,
        "strength": strength,
        "confidence": confidence,
    }


def build_explanation(
    info: dict,
    df: pd.DataFrame,
    genus_effects: dict,
    pred_probs: dict,
    decision: dict,
) -> str:
    question = info.get("research_questions", [""])[0]

    # Basic descriptive stats
    genus_counts = df["genus"].value_counts().to_dict()
    overall_props = (
        df.groupby("genus")
        .apply(lambda g: float(g["num_amtl"].sum() / g["sockets"].sum()))
        .to_dict()
    )

    lines = []
    lines.append(
        f"Research question: {question}"
    )
    lines.append(
        "I modeled the proportion of antemortem tooth loss (number of missing teeth / total sockets) "
        "using a binomial regression (GLM with logit link). The predictors were genus, age at death, "
        "estimated probability of being male, and tooth class (anterior, premolar, posterior). "
        "Homo sapiens was treated as the reference genus."
    )
    lines.append(
        f"Sample sizes by genus (rows in the dataset): {genus_counts}."
    )
    lines.append(
        f"Raw AMTL proportions by genus (num_amtl / sockets, unadjusted): {overall_props}."
    )
    lines.append(
        f"Model-based predicted AMTL probabilities at mean age and sex and the most common tooth class: {pred_probs}."
    )
    if genus_effects:
        lines.append(
            "Genus effects (log-odds relative to Homo sapiens) from the regression "
            f"(negative values mean lower AMTL than humans): {genus_effects}."
        )

    lines.append(
        f"Based on these results, the answer to the question "
        f"\"Do modern humans have higher frequencies of AMTL compared to non-human primate genera "
        f"(Pan, Pongo, Papio), after accounting for age, sex, and tooth class?\" is: "
        f"{decision['response']}."
    )

    if decision["response"] == "Yes":
        lines.append(
            "The regression indicates that non-human genera have lower or comparable AMTL frequencies "
            "than humans once age, sex, and tooth class are controlled for, and the predicted probabilities "
            "are higher for humans than for the non-human genera considered."
        )
    else:
        lines.append(
            "The regression does not provide consistent evidence that humans have higher AMTL frequencies "
            "than all non-human genera once age, sex, and tooth class are controlled for, and any observed "
            "differences may be small or statistically uncertain."
        )

    lines.append(
        "Strength reflects how strongly the estimated effects favor the chosen answer (magnitude and "
        "consistency of genus coefficients and predicted probabilities), whereas confidence reflects my "
        "trust in the conclusion given model assumptions, sampling variation, and potential unmodeled "
        "differences between populations."
    )

    return "\n".join(lines)


def write_conclusion(path: Path, decision: dict, explanation: str) -> None:
    output = {
        "response": decision["response"],
        "strength": decision["strength"],
        "confidence": decision["confidence"],
        "explanation": explanation,
    }
    with path.open("w") as f:
        json.dump(output, f)


def main() -> None:
    base_path = Path(__file__).resolve().parent

    info, df_raw = load_data(base_path)
    df = clean_data(df_raw)

    model = fit_model(df)
    genus_effects = summarize_genus_effects(model)
    pred_probs = compute_predicted_probs_by_genus(df, model)
    decision = decide_conclusion(genus_effects, pred_probs)
    explanation = build_explanation(info, df, genus_effects, pred_probs, decision)

    write_conclusion(base_path / "conclusion.txt", decision, explanation)


if __name__ == "__main__":
    main()
