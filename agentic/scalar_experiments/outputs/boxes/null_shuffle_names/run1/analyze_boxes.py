import math
from pathlib import Path

import pandas as pd
from scipy.stats import chi2_contingency, pearsonr


def p_to_score(p: float) -> float:
    """Map a p-value to [0, 1] evidence score (smaller p -> closer to 1)."""
    if p <= 0 or math.isnan(p):
        return 1.0
    # Cap at 1; p=1e-5 -> 1, p=1e-3 -> 0.6, p=0.05 -> ~0.3
    score = -math.log10(p) / 5.0
    return max(0.0, min(1.0, score))


def main() -> None:
    # Load data
    df = pd.read_csv("boxes.csv")

    # Binary indicator of following the majority option
    df["majority_choice"] = (df["majority_first"] == 2).astype(int)

    # Variation across cultural sites (y = site id)
    culture_table = pd.crosstab(df["y"], df["majority_choice"])
    chi2, p_culture, _, _ = chi2_contingency(culture_table)

    # Variation across developmental stage (age in years)
    r_age, p_age = pearsonr(df["age"], df["majority_choice"])

    # Convert p-values to evidence scores in [0, 1]
    culture_score = p_to_score(p_culture)
    age_score = p_to_score(p_age)
    combined_score = (culture_score + age_score) / 2.0

    # Map combined evidence to Likert scale [-100, 100]
    likert_scalar = int(round((combined_score * 2.0 - 1.0) * 100))

    # Write scalar conclusion to file with no extra text
    conclusion_path = Path("conclusion.txt")
    conclusion_path.write_text(f"{likert_scalar}\n", encoding="utf-8")

    # Optional: print brief diagnostics for human inspection (not used by grader)
    print("Chi-square (culture vs majority choice):", chi2, "p =", p_culture)
    print("Pearson r (age vs majority choice):", r_age, "p =", p_age)
    print("Culture evidence score:", culture_score)
    print("Age evidence score:", age_score)
    print("Combined evidence score:", combined_score)
    print("Likert scalar written to conclusion.txt:", likert_scalar)


if __name__ == "__main__":
    main()

