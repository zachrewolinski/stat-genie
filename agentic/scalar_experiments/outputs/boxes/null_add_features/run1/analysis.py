import pandas as pd
from scipy.stats import chi2_contingency


def compute_dimension_stats(df, group_col):
    """
    Compute chi-square p-value and range of majority-choice proportions
    across levels of group_col.
    """
    df = df.copy()
    df["majority_choice"] = (df["y"] == 2).astype(int)

    ct = pd.crosstab(df[group_col], df["majority_choice"])

    # Ensure both outcome levels are present to avoid key errors
    if 1 not in ct.columns:
        # No majority choices at all
        props = pd.Series(0.0, index=ct.index)
    else:
        props = ct[1] / ct.sum(axis=1)

    # Chi-square test of independence
    chi2, p_val, dof, expected = chi2_contingency(ct)
    range_prop = float(props.max() - props.min())

    return {
        "p_val": float(p_val),
        "range_prop": range_prop,
        "props": props,
        "table": ct,
    }


def dimension_score(range_val: float, p_val: float) -> float:
    """
    Map variation magnitude and significance into a [-1, 1] strength score.
    Higher is stronger evidence that majority preference varies with the factor.
    """
    # Cap range at 0.30 (30 percentage points) for scaling
    capped_range = max(min(range_val, 0.30), 0.0)
    range_scale = capped_range / 0.30  # 0..1

    if p_val < 1e-6:
        sig_scale = 1.0
    elif p_val < 1e-3:
        sig_scale = 0.8
    elif p_val < 1e-2:
        sig_scale = 0.6
    elif p_val < 5e-2:
        sig_scale = 0.4
    elif p_val < 1e-1:
        sig_scale = 0.2
    else:
        sig_scale = 0.0

    combo = 0.6 * range_scale + 0.4 * sig_scale

    # If variation is extremely small and there is little evidence of dependence,
    # treat this as weak evidence for invariance (slightly negative score).
    if range_val < 0.05 and p_val > 0.5:
        combo = -0.2 * (1.0 - min(p_val, 1.0))

    # Clamp to [-1, 1]
    return max(min(combo, 1.0), -1.0)


def main():
    df = pd.read_csv("boxes.csv")

    # Overall majority-following tendency
    df["majority_choice"] = (df["y"] == 2).astype(int)
    overall_majority_prop = float(df["majority_choice"].mean())

    # Variation across cultures
    culture_stats = compute_dimension_stats(df, "culture")
    culture_score = dimension_score(culture_stats["range_prop"], culture_stats["p_val"])

    # Variation across age groups (developmental stages)
    age_stats = compute_dimension_stats(df, "age")
    age_score = dimension_score(age_stats["range_prop"], age_stats["p_val"])

    # Combine culture and age evidence equally
    combined_score = 0.5 * (culture_score + age_score)
    combined_score = max(min(combined_score, 1.0), -1.0)
    likert_scalar = int(round(combined_score * 100))

    # Print summary for inspection
    print("Overall majority-following proportion:", overall_majority_prop)
    print("Culture majority-choice range:", culture_stats["range_prop"])
    print("Culture chi-square p-value:", culture_stats["p_val"])
    print("Age majority-choice range:", age_stats["range_prop"])
    print("Age chi-square p-value:", age_stats["p_val"])
    print("Culture score ([-1,1]):", culture_score)
    print("Age score ([-1,1]):", age_score)
    print("Combined score ([-1,1]):", combined_score)
    print("LIKERT_SCALAR:", likert_scalar)

    # Write scalar conclusion to file as required.
    with open("conclusion.txt", "w", encoding="utf-8") as f:
        f.write(str(likert_scalar))


if __name__ == "__main__":
    main()

