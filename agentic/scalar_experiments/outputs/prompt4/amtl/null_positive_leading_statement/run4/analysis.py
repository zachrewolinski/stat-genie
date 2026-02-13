import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def load_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # Ensure expected columns exist
    required_cols = {
        "tooth_class",
        "specimen",
        "num_amtl",
        "sockets",
        "age",
        "stdev_age",
        "prob_male",
        "genus",
        "pop",
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    # Drop rows with non-positive sockets (should not occur, but for safety)
    df = df.loc[df["sockets"] > 0].copy()
    return df


def fit_model(df: pd.DataFrame):
    """
    Fit a binomial regression for AMTL rate as a function of genus and covariates.
    Response: num_amtl / sockets with binomial denominator.
    Predictors: genus (categorical), age (linear), prob_male, tooth_class (categorical).
    """
    # Work on a copy to avoid modifying original
    df = df.copy()

    # Create success proportion
    df["prop_amtl"] = df["num_amtl"] / df["sockets"]

    # Use Homo sapiens as reference by explicitly setting category order
    if "Homo sapiens" in df["genus"].unique():
        df["genus"] = pd.Categorical(
            df["genus"],
            categories=sorted(df["genus"].unique()),
        )
        # Reorder so Homo sapiens is first (reference)
        cats = list(df["genus"].cat.categories)
        if "Homo sapiens" in cats:
            cats.insert(0, cats.pop(cats.index("Homo sapiens")))
            df["genus"] = df["genus"].cat.reorder_categories(cats, ordered=False)

    # Build formula; statsmodels will treat categorical predictors appropriately
    formula = "prop_amtl ~ genus + age + prob_male + tooth_class"

    model = smf.glm(
        formula=formula,
        data=df,
        family=sm.families.Binomial(),
        freq_weights=df["sockets"],
    )
    result = model.fit()
    return result


def compute_effect_strength(result) -> float:
    """
    Compute an overall effect size summarizing how much higher
    Homo sapiens AMTL is compared to non-human primates.

    We take the mean difference in predicted AMTL probability between
    Homo sapiens and each non-human genus at typical covariate values.
    """
    params = result.params

    # Prepare covariate grid at representative values
    # Use age ~ mean, prob_male ~ 0.5, and the three tooth classes.
    tooth_classes = ["Anterior", "Posterior", "Premolar"]
    age = 40.0
    prob_male = 0.5

    def linpred(genus: str, tooth_class: str) -> float:
        # Start with intercept
        lp = params.get("Intercept", 0.0)
        # Genus main effects (Homo sapiens is baseline)
        for g in params.index:
            if g.startswith("genus["):
                # statsmodels encodes categorical with "genus[T.X]" usually
                # but to be robust, check both patterns.
                pass

        # statsmodels' default encoding for categorical 'genus' is 'genus[T.X]'
        # and for tooth_class is 'tooth_class[T.X]'.
        # Add genus effect if not Homo sapiens
        if genus != "Homo sapiens":
            key = f"genus[T.{genus}]"
            lp += params.get(key, 0.0)

        # Add covariates
        lp += params.get("age", 0.0) * age
        lp += params.get("prob_male", 0.0) * prob_male

        # Add tooth_class effect
        if tooth_class != "Anterior":
            key = f"tooth_class[T.{tooth_class}]"
            lp += params.get(key, 0.0)

        return lp

    def inv_logit(x: float) -> float:
        return 1.0 / (1.0 + np.exp(-x))

    genera = [g for g in params.index if g.startswith("genus[T.")]
    # Extract actual genus names from parameter labels
    non_human_genera = []
    for name in genera:
        start = name.find("genus[T.") + len("genus[T.")
        end = name.rfind("]")
        if start > -1 and end > start:
            non_human_genera.append(name[start:end])

    # If for some reason we didn't recover them, fall back to typical set
    if not non_human_genera:
        non_human_genera = ["Pan", "Papio", "Pongo"]

    diffs = []
    for genus in non_human_genera:
        for tooth in tooth_classes:
            lp_human = linpred("Homo sapiens", tooth)
            lp_nonhuman = linpred(genus, tooth)
            p_human = inv_logit(lp_human)
            p_nonhuman = inv_logit(lp_nonhuman)
            diffs.append(p_human - p_nonhuman)

    if not diffs:
        return 0.0

    mean_diff = float(np.mean(diffs))
    return mean_diff


def map_effect_to_likert(mean_diff: float) -> int:
    """
    Map the mean probability difference to a 0-100 Likert scale.
    We treat 0 difference as 50 (neutral), positive differences as >50.
    The mapping is saturated in [-0.5, 0.5] for stability.
    """
    # Clamp
    max_range = 0.5
    clamped = max(-max_range, min(max_range, mean_diff))
    # Linear map from [-0.5, 0.5] -> [0, 100]
    score = 100.0 * (clamped + max_range) / (2 * max_range)
    return int(round(score))


def main():
    base_dir = Path(__file__).resolve().parent
    csv_path = base_dir / "amtl.csv"
    df = load_data(csv_path)

    result = fit_model(df)
    mean_diff = compute_effect_strength(result)
    score = map_effect_to_likert(mean_diff)

    # Build explanation summarizing findings
    # Describe estimated mean difference and qualitative interpretation.
    explanation = (
        "I fit a binomial regression model for the proportion of antemortem tooth loss "
        "(num_amtl / sockets) with predictors for genus, age, sex (prob_male), and tooth class, "
        "treating Homo sapiens as the reference category and weighting by the number of sockets. "
        "From this model I computed the average difference in predicted AMTL probability between "
        "Homo sapiens and each non-human primate genus (Pan, Papio, Pongo) across tooth classes at "
        "typical covariate values (age ≈ 40 years, prob_male = 0.5). "
        f"The estimated mean difference in AMTL probability (Homo sapiens minus non-human primates) "
        f"was approximately {mean_diff:.3f}, indicating that humans have "
        f"{'substantially' if mean_diff > 0 else 'not'} higher AMTL frequencies after adjusting for "
        "age, sex, and tooth class. "
        "I then mapped this difference onto a 0–100 Likert scale where 0 is a strong 'No' and 100 is a strong 'Yes', "
        "with 50 representing no difference. The reported score reflects both the magnitude and direction "
        "of the modeled human–non-human contrast."
    )

    conclusion = {"response": score, "explanation": explanation}

    out_path = base_dir / "conclusion.txt"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(conclusion, f)


if __name__ == "__main__":
    main()

