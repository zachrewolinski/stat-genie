import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


DATA_FILE = Path("amtl.csv")
CONCLUSION_FILE = Path("conclusion.txt")


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Proportion of missing teeth within tooth class
    df["amtl_prop"] = df["feature3"] / df["feature4"]
    return df


def fit_binomial_model(df: pd.DataFrame):
    # Binomial regression of AMTL proportion with number of observable sockets as weights
    formula = "amtl_prop ~ C(feature8) + feature5 + feature7 + C(feature1)"
    model = smf.glm(
        formula=formula,
        data=df,
        family=sm.families.Binomial(),
        freq_weights=df["feature4"],
    )
    result = model.fit()
    return result


def genus_effects_summary(result) -> dict:
    """
    Summarize genus effects relative to Homo sapiens.
    statsmodels chooses a reference category for C(feature8);
    we assume Homo sapiens is the baseline as it is the most common
    and typically the first category.
    """
    coefs = result.params
    ses = result.bse
    # Collect coefficients for non-human genera relative to Homo sapiens
    genus_effects = {}
    for name, coef in coefs.items():
        if name.startswith("C(feature8)[T."):
            genus = name.split("[T.")[-1].rstrip("]")
            se = ses[name]
            z = coef / se if se > 0 else np.nan
            genus_effects[genus] = {
                "coef": float(coef),
                "se": float(se),
                "z": float(z),
                "pvalue": float(result.pvalues[name]),
            }
    return genus_effects


def answer_research_question(genus_effects: dict) -> tuple[str, int, str]:
    """
    Determine whether modern humans have higher AMTL frequencies than
    non-human primates after accounting for age, sex, and tooth class.
    Since Homo sapiens is treated as the reference,
    negative coefficients for non-human genera indicate lower AMTL
    than humans, and positive coefficients indicate higher AMTL.
    """
    # Track how many genera clearly have lower / higher AMTL than humans
    lower_genera = []
    higher_genera = []
    ambiguous_genera = []

    for genus, stats in genus_effects.items():
        coef = stats["coef"]
        pvalue = stats["pvalue"]
        if pvalue < 0.05:
            if coef < 0:
                lower_genera.append(genus)
            elif coef > 0:
                higher_genera.append(genus)
            else:
                ambiguous_genera.append(genus)
        else:
            ambiguous_genera.append(genus)

    # Decide on binary answer
    if higher_genera and not lower_genera:
        response = "No"
    elif lower_genera and not higher_genera:
        response = "Yes"
    else:
        # Mixed or ambiguous pattern
        response = "No"

    # Confidence heuristic:
    # - Start from moderate confidence
    confidence = 60
    if response == "Yes":
        # Increase confidence if all three non-human genera show
        # significantly lower AMTL than humans
        if len(lower_genera) >= 3 and not higher_genera and not ambiguous_genera:
            confidence = 90
        elif lower_genera and not higher_genera:
            confidence = 80
    else:
        if higher_genera and not lower_genera and not ambiguous_genera:
            confidence = 85

    # Build explanation string
    explanation_parts = [
        "I fit a binomial regression model predicting the proportion of antemortem tooth loss (AMTL) within each tooth class as a function of genus, estimated age at death, estimated sex, and tooth class, using the number of observable sockets as binomial weights."
    ]

    if genus_effects:
        descs = []
        for genus, stats in genus_effects.items():
            direction = "lower" if stats["coef"] < 0 else "higher"
            descs.append(
                f"{genus} shows {direction} AMTL log-odds relative to Homo sapiens (coefficient={stats['coef']:.3f}, p={stats['pvalue']:.3f})"
            )
        explanation_parts.append(" ".join(descs))

    if response == "Yes":
        explanation_parts.append(
            "Across the non-human primate genera, the model indicates that their AMTL frequencies are generally lower than those of modern humans after adjusting for age, sex, and tooth class."
        )
    else:
        explanation_parts.append(
            "The model does not provide consistent evidence that modern humans have higher AMTL frequencies than all non-human primate genera after adjusting for covariates."
        )

    explanation = " ".join(explanation_parts)
    return response, confidence, explanation


def write_conclusion(response: str, confidence: int, explanation: str, path: Path):
    obj = {
        "response": response,
        "confidence": int(confidence),
        "explanation": explanation,
    }
    path.write_text(json.dumps(obj, ensure_ascii=False))


def main():
    df = load_data(DATA_FILE)
    result = fit_binomial_model(df)
    genus_effects = genus_effects_summary(result)
    response, confidence, explanation = answer_research_question(genus_effects)
    write_conclusion(response, confidence, explanation, CONCLUSION_FILE)


if __name__ == "__main__":
    main()

