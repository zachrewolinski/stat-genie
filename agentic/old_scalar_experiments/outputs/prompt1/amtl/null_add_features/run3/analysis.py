import json
from pathlib import Path

import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


DATA_FILE = Path("amtl.csv")
INFO_FILE = Path("info.json")
OUTPUT_FILE = Path("conclusion.txt")


def load_metadata() -> dict:
    """Load metadata, including the research question, from info.json if available."""
    if INFO_FILE.exists():
        with INFO_FILE.open("r") as f:
            return json.load(f)
    return {}


def load_data() -> pd.DataFrame:
    """Load the AMTL dataset from CSV."""
    df = pd.read_csv(DATA_FILE)
    return df


def prepare_model_data(df: pd.DataFrame) -> pd.DataFrame:
    """Filter and prepare data for binomial regression."""
    required_cols = [
        "num_amtl",
        "sockets",
        "age",
        "prob_male",
        "tooth_class",
        "genus",
    ]
    df_model = df[required_cols].dropna().copy()

    # Restrict to the taxa relevant for the research question
    target_taxa = {"Homo sapiens", "Pan", "Pongo", "Papio"}
    df_model = df_model[df_model["genus"].isin(target_taxa)].copy()

    # Remove any rows with non-positive socket counts
    df_model = df_model[df_model["sockets"] > 0].copy()

    # Proportion of missing teeth to be modeled with binomial GLM
    df_model["prop_amtl"] = df_model["num_amtl"] / df_model["sockets"]

    return df_model


def fit_binomial_glm(df_model: pd.DataFrame):
    """
    Fit a binomial GLM for AMTL proportion with genus, age, sex, and tooth class.

    We use Homo sapiens as the reference genus so that non-human genus coefficients
    reflect the difference in log-odds relative to modern humans.
    """
    # Define model formula with explicit reference category for genus
    formula = (
        "prop_amtl ~ C(genus, Treatment(reference='Homo sapiens'))"
        " + age + prob_male + C(tooth_class)"
    )

    # Use sockets as frequency weights so that each observation represents
    # multiple Bernoulli trials (one per tooth socket)
    model = smf.glm(
        formula=formula,
        data=df_model,
        family=sm.families.Binomial(),
        freq_weights=df_model["sockets"].to_numpy(),
    )
    result = model.fit()
    return result


def summarize_genus_effects(result) -> dict:
    """
    Extract effects for non-human genera relative to Homo sapiens.

    Returns a dict keyed by genus with coefficient, CI, and p-value.
    """
    params = result.params
    pvalues = result.pvalues
    conf_int = result.conf_int()

    genus_effects = {}
    for name in params.index:
        if "genus" in name and "[T." in name:
            # Parameter names look like: C(genus, Treatment(reference='Homo sapiens'))[T.Papio]
            genus_label = name.split("[T.", 1)[-1].rstrip("]")
            coef = float(params[name])
            pval = float(pvalues[name])
            ci_low = float(conf_int.loc[name, 0])
            ci_high = float(conf_int.loc[name, 1])
            genus_effects[genus_label] = {
                "coef": coef,
                "pvalue": pval,
                "ci_low": ci_low,
                "ci_high": ci_high,
            }
    return genus_effects


def predict_amtl_by_genus(result, df_model: pd.DataFrame) -> dict:
    """
    Compute adjusted AMTL proportions for each genus at typical covariate values.

    We hold age and sex at their overall means and tooth class at the most
    common category to obtain adjusted predictions.
    """
    mean_age = float(df_model["age"].mean())
    mean_prob_male = float(df_model["prob_male"].mean())
    common_tooth_class = df_model["tooth_class"].mode().iloc[0]

    genera = sorted(df_model["genus"].unique())
    pred_rows = []
    for g in genera:
        pred_rows.append(
            {
                "genus": g,
                "age": mean_age,
                "prob_male": mean_prob_male,
                "tooth_class": common_tooth_class,
            }
        )
    pred_df = pd.DataFrame(pred_rows)
    predictions = result.get_prediction(pred_df)
    summary_frame = predictions.summary_frame()

    # summary_frame["mean"] is the predicted AMTL proportion on the response scale
    genus_preds = {}
    for g, mean_pred in zip(genera, summary_frame["mean"].to_numpy()):
        genus_preds[g] = float(mean_pred)

    return genus_preds


def determine_answer(genus_effects: dict) -> (str, str):
    """
    Decide whether humans have higher AMTL than non-human genera.

    Because coefficients represent (genus - Homo sapiens) on the log-odds scale,
    humans have higher AMTL if every non-human genus has a significantly
    negative coefficient (CI entirely below zero).
    """
    # Extract only the non-human genera of interest
    nonhuman_targets = ["Pan", "Pongo", "Papio"]
    missing = [g for g in nonhuman_targets if g not in genus_effects]

    if missing:
        explanation = (
            "The regression model could not estimate genus effects for all "
            "non-human genera of interest (missing: "
            + ", ".join(missing)
            + "), so I cannot conclude that humans have higher AMTL frequencies "
            "than each of Pan, Pongo, and Papio after adjustment."
        )
        return "No", explanation

    significant_and_lower = []
    for g in nonhuman_targets:
        eff = genus_effects[g]
        coef = eff["coef"]
        pval = eff["pvalue"]
        ci_low = eff["ci_low"]
        ci_high = eff["ci_high"]
        is_lower = (coef < 0.0) and (ci_high < 0.0) and (pval < 0.05)
        significant_and_lower.append((g, is_lower, coef, ci_low, ci_high, pval))

    humans_higher_all = all(item[1] for item in significant_and_lower)

    if humans_higher_all:
        response = "Yes"
    else:
        response = "No"

    # Build a concise textual summary of the evidence
    parts = []
    for g, is_lower, coef, ci_low, ci_high, pval in significant_and_lower:
        if is_lower:
            desc = (
                f"For {g}, the log-odds of AMTL are lower than for Homo sapiens "
                f"(coef={coef:.3f}, 95% CI [{ci_low:.3f}, {ci_high:.3f}], "
                f"p={pval:.3g})."
            )
        else:
            desc = (
                f"For {g}, the difference in AMTL relative to Homo sapiens is "
                f"not clearly lower (coef={coef:.3f}, 95% CI [{ci_low:.3f}, "
                f"{ci_high:.3f}], p={pval:.3g})."
            )
        parts.append(desc)

    explanation = " ".join(parts)

    if response == "Yes":
        explanation = (
            "Using a binomial logistic regression of AMTL proportion on genus, "
            "age, sex (probability of being male), and tooth class, with Homo "
            "sapiens as the reference genus and sockets as the number of trials, "
            "the model indicates that humans have higher AMTL frequencies than "
            "the non-human primate genera after adjustment. "
        ) + explanation
    else:
        explanation = (
            "A binomial logistic regression of AMTL proportion on genus, age, "
            "sex (probability of being male), and tooth class, with Homo "
            "sapiens as the reference genus and sockets as the number of trials, "
            "does not provide strong evidence that humans have higher AMTL "
            "frequencies than each non-human primate genus after adjustment. "
        ) + explanation

    return response, explanation


def main() -> None:
    # Load metadata and data (research question is used mainly for context)
    _ = load_metadata()
    df = load_data()
    df_model = prepare_model_data(df)

    if df_model.empty:
        result_obj = {
            "response": "No",
            "explanation": (
                "After filtering the dataset for relevant variables and genera, "
                "no data remained for modeling AMTL frequencies, so the research "
                "question cannot be answered with this dataset."
            ),
        }
        OUTPUT_FILE.write_text(json.dumps(result_obj))
        return

    result = fit_binomial_glm(df_model)
    genus_effects = summarize_genus_effects(result)
    genus_preds = predict_amtl_by_genus(result, df_model)

    response, explanation = determine_answer(genus_effects)

    # Append information about adjusted predicted AMTL proportions by genus
    if genus_preds:
        ordered_genera = sorted(genus_preds.keys())
        pred_parts = [
            f"{g}: {genus_preds[g]*100:.1f}%" for g in ordered_genera
        ]
        explanation = (
            explanation
            + " Adjusted predicted AMTL frequencies (at mean age and sex and "
            "typical tooth class) are approximately: "
            + "; ".join(pred_parts)
            + "."
        )

    result_obj = {
        "response": response,
        "explanation": explanation,
    }
    OUTPUT_FILE.write_text(json.dumps(result_obj))


if __name__ == "__main__":
    main()
