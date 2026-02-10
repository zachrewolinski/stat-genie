import numpy as np
import pandas as pd
from scipy import stats


def main() -> None:
    # Load data
    df = pd.read_csv("caschools.csv")

    # Define key variables
    # Student–teacher ratio: higher values mean more students per teacher.
    df["stratio"] = df["students"] / df["teachers"]

    # Academic performance proxy: average of reading and math scores.
    df["testscr"] = (df["read"] + df["math"]) / 2.0

    stratio = df["stratio"].to_numpy()
    testscr = df["testscr"].to_numpy()

    # Drop any rows with missing values in these fields (if any).
    mask = np.isfinite(stratio) & np.isfinite(testscr)
    stratio = stratio[mask]
    testscr = testscr[mask]

    # Compute Pearson correlation between student–teacher ratio and test scores.
    r, pval = stats.pearsonr(stratio, testscr)

    # Map statistical evidence to a Likert-scale scalar in [-100, 100].
    # The research question is:
    # "Is a lower student-teacher ratio associated with higher academic performance?"
    #
    # Higher student–teacher ratio implies more students per teacher, so
    # a *negative* correlation (r < 0) supports the hypothesis.
    #
    # We build a score that:
    #   - Is 0 if the association is not statistically significant (p >= 0.05).
    #   - Grows in magnitude with both |r| and statistical significance.
    #   - Is positive when evidence supports the research question (r < 0),
    #     and negative when evidence goes against it (r > 0).
    if pval >= 0.05:
        weight = 0.0
    else:
        weight = 1.0 - (pval / 0.05)
        weight = float(np.clip(weight, 0.0, 1.0))

    score_float = -100.0 * r * weight
    score = int(np.rint(np.clip(score_float, -100.0, 100.0)))

    # Print key diagnostics for transparency when the script is run.
    print(f"Pearson r(stratio, testscr) = {r:.4f}, p-value = {pval:.4g}")
    print(f"Derived Likert-scale conclusion score = {score}")

    # Write the final scalar conclusion to file as required.
    with open("conclusion.txt", "w", encoding="utf-8") as f:
        f.write(str(score))


if __name__ == "__main__":
    main()

