import pandas as pd
import numpy as np


def compute_scalar_from_data(df: pd.DataFrame) -> int:
    """
    Map evidence about children's reliance on majority social information
    to a Likert-style scalar in [-100, 100].
    """
    # feature1: 1=undemonstrated, 2=majority, 3=minority
    total = len(df)
    if total == 0:
        return 0

    majority = (df["feature1"] == 2).mean()
    minority = (df["feature1"] == 3).mean()
    undemo = (df["feature1"] == 1).mean()

    # Basic cross-cultural / developmental sanity checks:
    # feature5: site (culture), feature3: age
    sites = df["feature5"].unique()
    per_site_majority = df.groupby("feature5")["feature1"].apply(lambda s: (s == 2).mean())
    per_age_majority = df.groupby("feature3")["feature1"].apply(lambda s: (s == 2).mean())

    # Aggregate indicators
    majority_overall_bias = majority - max(minority, undemo)

    # Cross-cultural consistency: how many sites show majority > max(minority, undemo)?
    site_majority_adv = 0
    for site, sub in df.groupby("feature5"):
        m = (sub["feature1"] == 2).mean()
        other = max(
            (sub["feature1"] == 3).mean(),
            (sub["feature1"] == 1).mean(),
        )
        if m > other:
            site_majority_adv += 1

    # Developmental trend: correlation between age and majority choice
    age_corr = np.corrcoef(df["feature3"], (df["feature1"] == 2).astype(float))[0, 1]

    # Scoring heuristic:
    score = 0.0

    # Strong positive if majority is clearly preferred overall
    score += 120 * majority_overall_bias  # maps ~0.4 advantage -> ~48 points

    # Add cross-cultural consistency: proportion of sites with majority advantage
    if len(sites) > 0:
        score += 40 * (site_majority_adv / len(sites) - 0.5)  # centered at 0.5

    # Add developmental trend: positive correlation supports "develops with age"
    score += 40 * age_corr

    # Clip to [-100, 100] and return as int
    score = int(max(-100, min(100, round(score))))
    return score


def main() -> None:
    df = pd.read_csv("boxes.csv")
    scalar = compute_scalar_from_data(df)
    with open("conclusion.txt", "w") as f:
        f.write(str(scalar))


if __name__ == "__main__":
    main()

