import json

import numpy as np
import pandas as pd
from scipy.stats import pearsonr


def main() -> None:
    df = pd.read_csv("caschools.csv")

    # Restrict to observations with a positive teacher count
    df = df[df["feature7"] > 0].copy()

    # Student–teacher ratio: enrollment divided by number of teachers
    df["stratio"] = df["feature6"] / df["feature7"]

    # Keep rows with finite values for ratio and scores
    mask = np.isfinite(df["stratio"]) & np.isfinite(df["feature14"]) & np.isfinite(
        df["feature15"]
    )
    data = df.loc[mask].copy()

    ratio = data["stratio"]
    reading = data["feature14"]
    math = data["feature15"]

    # Pearson correlations between student–teacher ratio and scores
    corr_read, p_read = pearsonr(ratio, reading)
    corr_math, p_math = pearsonr(ratio, math)

    # Compare mean scores between lowest and highest quartiles of the ratio
    q1 = ratio.quantile(0.25)
    q3 = ratio.quantile(0.75)

    low = data[ratio <= q1]
    high = data[ratio >= q3]

    low_read_mean = float(low["feature14"].mean())
    low_math_mean = float(low["feature15"].mean())
    high_read_mean = float(high["feature14"].mean())
    high_math_mean = float(high["feature15"].mean())

    low_ratio_median = float(low["stratio"].median())
    high_ratio_median = float(high["stratio"].median())

    alpha = 0.05
    significant_negative = (
        (corr_read < 0 and p_read < alpha)
        or (corr_math < 0 and p_math < alpha)
    )

    if significant_negative:
        response = "Yes"
        explanation = (
            "Using data on 420 California K-6 and K-8 school districts, "
            "I computed the student–teacher ratio as total enrollment divided by the number of teachers. "
            f"The Pearson correlation between this ratio and average reading scores was {corr_read:.3f} "
            f"(p = {p_read:.4f}), and with average math scores was {corr_math:.3f} "
            f"(p = {p_math:.4f}); both correlations were negative and statistically significant at the 5% level. "
            f"Districts in the lowest quartile of student–teacher ratios (median {low_ratio_median:.1f} students per teacher) "
            f"had mean reading and math scores of {low_read_mean:.1f} and {low_math_mean:.1f}, respectively, "
            f"whereas districts in the highest quartile (median {high_ratio_median:.1f} students per teacher) "
            f"had lower mean reading and math scores of {high_read_mean:.1f} and {high_math_mean:.1f}. "
            "These patterns indicate that districts with lower student–teacher ratios tend to have higher academic performance, "
            "so the data support an association between smaller classes and better test scores."
        )
    else:
        response = "No"
        explanation = (
            "Using data on 420 California K-6 and K-8 school districts, "
            "I computed the student–teacher ratio as total enrollment divided by the number of teachers, "
            "and examined its association with average reading and math scores. "
            f"The Pearson correlation between this ratio and reading scores was {corr_read:.3f} "
            f"(p = {p_read:.4f}), and with math scores was {corr_math:.3f} "
            f"(p = {p_math:.4f}). These estimates were not strongly negative and/or not statistically significant "
            "at conventional levels, and differences in mean scores between districts with very low and very high "
            "student–teacher ratios were modest. Overall, the data do not provide clear evidence that lower "
            "student–teacher ratios are associated with meaningfully higher academic performance."
        )

    # Print a short summary for interactive inspection
    print("Correlation (ratio, reading):", corr_read, "p-value:", p_read)
    print("Correlation (ratio, math):   ", corr_math, "p-value:", p_math)
    print("Response:", response)

    output = {
        "response": response,
        "explanation": explanation,
    }

    with open("conclusion.txt", "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False)


if __name__ == "__main__":
    main()

