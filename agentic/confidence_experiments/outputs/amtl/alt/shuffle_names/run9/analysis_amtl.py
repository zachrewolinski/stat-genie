import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def load_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    # Rename columns to clearer semantic names based on info.json metadata
    df = df.rename(
        columns={
            "sockets": "tooth_type",  # Anterior/Posterior/Premolar
            "prob_male": "specimen_id",
            "genus": "num_missing",  # number of missing teeth of given class
            "age": "num_sockets",  # observable sockets
            "pop": "age_at_death",
            "num_amtl": "age_uncertainty",
            "stdev_age": "prob_male",  # estimate of sex (probability male)
            "tooth_class": "genus",  # Homo sapiens, Pan, Papio, Pongo
        }
    )

    # Basic cleaning
    df["num_missing"] = pd.to_numeric(df["num_missing"], errors="coerce")
    df["num_sockets"] = pd.to_numeric(df["num_sockets"], errors="coerce")
    df["age_at_death"] = pd.to_numeric(df["age_at_death"], errors="coerce")
    df["prob_male"] = pd.to_numeric(df["prob_male"], errors="coerce")

    # Drop rows with missing key values or invalid socket counts
    df = df.dropna(subset=["num_missing", "num_sockets", "age_at_death", "prob_male", "genus", "tooth_type"])
    df = df[df["num_sockets"] > 0]

    # Proportion of missing teeth and human indicator
    df["prop_missing"] = df["num_missing"] / df["num_sockets"]
    df["is_human"] = (df["genus"] == "Homo sapiens").astype(int)

    # Ensure categories
    df["genus"] = df["genus"].astype("category")
    df["tooth_type"] = df["tooth_type"].astype("category")

    return df


def fit_model(df: pd.DataFrame):
    # Binomial regression with logit link; use frequency weights for socket counts
    formula = "prop_missing ~ C(genus, Treatment(reference='Homo sapiens')) + age_at_death + prob_male + C(tooth_type)"
    model = smf.glm(
        formula=formula,
        data=df,
        family=sm.families.Binomial(),
        freq_weights=df["num_sockets"],
    )
    result = model.fit()
    return result


def summarize_human_vs_nonhuman(result) -> dict:
    params = result.params
    pvalues = result.pvalues

    # Coefficients for non-human genera relative to Homo sapiens
    genus_effects = {}
    for genus in ["Pan", "Papio", "Pongo"]:
        term = f"C(genus, Treatment(reference='Homo sapiens'))[T.{genus}]"
        if term in params.index:
            genus_effects[genus] = {
                "coef": float(params[term]),
                "pvalue": float(pvalues[term]),
            }

    # Interpret results: negative coefficient means lower AMTL than humans
    significant_lower = {
        g: eff for g, eff in genus_effects.items() if eff["pvalue"] < 0.05 and eff["coef"] < 0
    }
    significant_higher = {
        g: eff for g, eff in genus_effects.items() if eff["pvalue"] < 0.05 and eff["coef"] > 0
    }

    return {
        "genus_effects": genus_effects,
        "significant_lower": significant_lower,
        "significant_higher": significant_higher,
    }


def compute_likert(summary: dict) -> int:
    # Map evidence strength to 0–100 scale for "Humans have higher AMTL than non-human primates"
    sig_lower = summary["significant_lower"]
    sig_higher = summary["significant_higher"]

    # If any non-human genus has significantly higher AMTL than humans, that is evidence against the hypothesis
    if sig_higher:
        return 20

    # If at least two non-human genera show significantly lower AMTL, strong evidence that humans have higher AMTL
    if len(sig_lower) >= 2:
        return 85

    # If one genus significantly lower and others trending lower, moderate evidence
    if len(sig_lower) == 1:
        return 70

    # No significant genus effects: little evidence for difference
    return 40


def build_explanation(result, summary: dict, likert: int) -> str:
    lines = []
    lines.append(
        "I fit a binomial regression model for the proportion of missing teeth (number of missing teeth divided by the number of observable sockets) per specimen–tooth-class combination."
    )
    lines.append(
        "The model included genus (Homo sapiens, Pan, Papio, Pongo) as the main predictor of interest, and controlled for estimated age at death, estimated probability of being male, and tooth class (anterior, posterior, premolar)."
    )
    lines.append(
        "The regression was implemented as a GLM with a binomial family and logit link, using the number of sockets as frequency weights so that rows with more observable teeth contributed proportionally more information."
    )

    # Summarize genus effects
    genus_effects = summary["genus_effects"]
    for genus, eff in genus_effects.items():
        direction = "lower" if eff["coef"] < 0 else "higher"
        lines.append(
            f"Relative to modern humans (Homo sapiens), the coefficient for {genus} was {eff['coef']:.3f} (p = {eff['pvalue']:.3g}), indicating {direction} AMTL frequencies for {genus} at comparable age, sex, and tooth class."
        )

    if summary["significant_lower"] and not summary["significant_higher"]:
        lines.append(
            "Across non-human primate genera, the statistically significant coefficients are negative, meaning that non-human primates tend to have lower AMTL frequencies than humans after adjusting for covariates."
        )
    elif summary["significant_higher"]:
        lines.append(
            "At least one non-human primate genus shows a significantly positive coefficient, indicating higher AMTL frequencies than humans after adjustment."
        )
    else:
        lines.append(
            "Genus coefficients are not consistently statistically significant, so evidence for systematic differences in AMTL frequencies between humans and non-human primates is weak."
        )

    if likert >= 75:
        conclusion = (
            "Taken together, these results provide strong evidence that modern humans have higher frequencies of antemortem tooth loss than non-human primates, "
            "even after accounting for age, sex, and tooth class."
        )
    elif likert >= 55:
        conclusion = (
            "Overall, the results suggest that modern humans are more likely to exhibit antemortem tooth loss than non-human primates after adjusting for age, sex, and tooth class, "
            "but the evidence is only moderate in strength."
        )
    elif likert >= 45:
        conclusion = (
            "Overall, the model does not show clear, statistically robust differences between humans and non-human primates in AMTL frequency once age, sex, and tooth class are controlled for."
        )
    else:
        conclusion = (
            "Overall, the model suggests that non-human primates do not have lower AMTL frequencies than humans after adjusting for age, sex, and tooth class; "
            "if anything, humans appear less likely to experience AMTL, although the evidence is not uniformly strong across genera."
        )

    lines.append(conclusion)
    return " ".join(lines)


def main():
    base_dir = Path(__file__).resolve().parent
    csv_path = base_dir / "amtl.csv"

    df = load_data(csv_path)
    result = fit_model(df)
    summary = summarize_human_vs_nonhuman(result)
    likert = compute_likert(summary)
    explanation = build_explanation(result, summary, likert)

    conclusion = {"response": int(likert), "explanation": explanation}

    out_path = base_dir / "conclusion.txt"
    out_path.write_text(json.dumps(conclusion, ensure_ascii=False))


if __name__ == "__main__":
    main()

