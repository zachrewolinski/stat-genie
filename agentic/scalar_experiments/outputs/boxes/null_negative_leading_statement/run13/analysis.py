import pandas as pd
from scipy.stats import chi2_contingency


def main() -> None:
    # Load data
    df = pd.read_csv("boxes.csv")

    # Binary indicator for following the majority option
    df["majority_follow"] = (df["y"] == 2).astype(int)

    # Define age groups capturing developmental stages
    bins = [3, 6, 9, 12, 14]
    labels = ["4-6", "7-9", "10-12", "13-14"]
    df["age_group"] = pd.cut(df["age"], bins=bins, labels=labels, include_lowest=True)

    # Contingency tables
    age_ct = pd.crosstab(df["age_group"], df["majority_follow"])
    culture_ct = pd.crosstab(df["culture"], df["majority_follow"])

    # Chi-square tests of independence
    chi2_age, p_age, _, _ = chi2_contingency(age_ct)
    chi2_culture, p_culture, _, _ = chi2_contingency(culture_ct)

    # Majority-following rates and ranges across groups
    age_rates = age_ct.div(age_ct.sum(axis=1), axis=0)[1]
    culture_rates = culture_ct.div(culture_ct.sum(axis=1), axis=0)[1]
    range_age = float(age_rates.max() - age_rates.min())
    range_culture = float(culture_rates.max() - culture_rates.min())

    # Map statistical evidence to Likert scalar (-100 to 100)
    scalar = map_to_scalar(p_age, p_culture, range_age, range_culture)

    # Optional: print a brief summary for human inspection
    print("p_age:", p_age)
    print("p_culture:", p_culture)
    print("range_age:", range_age)
    print("range_culture:", range_culture)
    print("scalar:", scalar)

    # Write final scalar conclusion
    with open("conclusion.txt", "w", encoding="utf-8") as f:
        f.write(str(int(scalar)))


def map_to_scalar(p_age: float, p_culture: float, range_age: float, range_culture: float) -> int:
    """
    Convert evidence about variation across age and culture into a scalar.

    Positive values = evidence that preferences vary (answer "Yes").
    Negative values = evidence that preferences do not vary (answer "No").
    """
    p_min = min(p_age, p_culture)
    range_max = max(range_age, range_culture)

    # Strong evidence of variation: small p-value and substantial differences in rates
    if p_min < 0.001 and range_max >= 0.25:
        return 80
    if p_min < 0.01 and range_max >= 0.20:
        return 60
    if p_min < 0.05 and range_max >= 0.15:
        return 40

    # Weak or ambiguous evidence of variation
    if p_min < 0.1 and range_max >= 0.10:
        return 20

    # No strong evidence of variation: lean toward "No"
    if p_min > 0.5 and range_max < 0.05:
        return -60
    if p_min > 0.2 and range_max < 0.10:
        return -40

    # Inconclusive / mixed signals: stay near neutral
    return 0


if __name__ == "__main__":
    main()

