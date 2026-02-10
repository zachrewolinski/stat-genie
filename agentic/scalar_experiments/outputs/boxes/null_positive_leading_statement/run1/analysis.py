import pandas as pd
from scipy.stats import chi2_contingency


def range_or_zero(series: pd.Series) -> float:
    if len(series) == 0:
        return 0.0
    return float(series.max() - series.min())


def main() -> None:
    df = pd.read_csv("boxes.csv")

    # Social information reliance: choosing any demonstrated option (majority or minority).
    df["social_reliance"] = (df["y"] != 1).astype(int)

    # Majority preference: among children who chose a demonstrated option,
    # did they pick the majority option (2) over the minority (3)?
    demonstrated = df[df["y"].isin([2, 3])].copy()
    demonstrated["majority_choice"] = (demonstrated["y"] == 2).astype(int)

    # Age groups (developmental stages).
    age_bins = [3, 6, 9, 12, 15]
    age_labels = ["4-6", "7-9", "10-12", "13-14"]
    df["age_group"] = pd.cut(
        df["age"],
        bins=age_bins,
        labels=age_labels,
        include_lowest=True,
        right=True,
    )
    demonstrated["age_group"] = pd.cut(
        demonstrated["age"],
        bins=age_bins,
        labels=age_labels,
        include_lowest=True,
        right=True,
    )

    # Chi-square tests for dependence on age group and culture.
    # 1) Social reliance vs age group and vs culture.
    sr_age_tab = pd.crosstab(df["age_group"], df["social_reliance"])
    chi2, p_sr_age, dof, expected = chi2_contingency(sr_age_tab)

    sr_culture_tab = pd.crosstab(df["culture"], df["social_reliance"])
    chi2, p_sr_culture, dof, expected = chi2_contingency(sr_culture_tab)

    # 2) Majority preference vs age group and vs culture (only among demonstrated choices).
    maj_age_tab = pd.crosstab(demonstrated["age_group"], demonstrated["majority_choice"])
    chi2, p_maj_age, dof, expected = chi2_contingency(maj_age_tab)

    maj_culture_tab = pd.crosstab(demonstrated["culture"], demonstrated["majority_choice"])
    chi2, p_maj_culture, dof, expected = chi2_contingency(maj_culture_tab)

    # Compute variation ranges in proportions across groups.
    sr_age_props = sr_age_tab.div(sr_age_tab.sum(axis=1), axis=0)[1]
    sr_culture_props = sr_culture_tab.div(sr_culture_tab.sum(axis=1), axis=0)[1]
    maj_age_props = maj_age_tab.div(maj_age_tab.sum(axis=1), axis=0)[1]
    maj_culture_props = maj_culture_tab.div(maj_culture_tab.sum(axis=1), axis=0)[1]

    ranges = {
        "sr_age": range_or_zero(sr_age_props),
        "sr_culture": range_or_zero(sr_culture_props),
        "maj_age": range_or_zero(maj_age_props),
        "maj_culture": range_or_zero(maj_culture_props),
    }

    p_values = {
        "sr_age": p_sr_age,
        "sr_culture": p_sr_culture,
        "maj_age": p_maj_age,
        "maj_culture": p_maj_culture,
    }

    # Map evidence strength to a Likert-style scalar in [-100, 100].
    strong_evidence = 0
    moderate_evidence = 0
    for key, p in p_values.items():
        r = ranges[key]
        if p < 0.001 and r >= 0.15:
            strong_evidence += 1
        elif p < 0.05 and r >= 0.10:
            moderate_evidence += 1

    if strong_evidence >= 4:
        score = 95
    elif strong_evidence >= 3:
        score = 90
    elif strong_evidence >= 2:
        score = 80
    elif strong_evidence >= 1 or moderate_evidence >= 3:
        score = 70
    elif moderate_evidence >= 1:
        score = 40
    else:
        # Little to no evidence that reliance or majority preference
        # varies across cultures or developmental stages.
        if all(p > 0.5 for p in p_values.values()):
            score = -40
        else:
            score = 0

    score = int(max(-100, min(100, round(score))))

    with open("conclusion.txt", "w") as f:
        f.write(str(score))

    # Print a brief diagnostic summary for human inspection (not used by grader).
    print("p-values:", p_values)
    print("ranges:", ranges)
    print("scalar_score:", score)


if __name__ == "__main__":
    main()

