import pandas as pd
from scipy import stats
import statsmodels.api as sm


def main() -> None:
    df = pd.read_csv("affairs.csv")

    # Based on info.json descriptions, the "religiousness" column actually
    # encodes whether there are children in the marriage (yes/no),
    # and the "age" column encodes affair frequency over the past year.
    children_col = "religiousness"
    affairs_col = "age"

    # Keep only rows with non-missing values for these two columns.
    data = df[[children_col, affairs_col]].dropna()

    # Map children indicator: yes -> 1 (children), no -> 0 (no children).
    children_flag = data[children_col].astype(str).str.lower().map(
        {"yes": 1, "no": 0}
    )
    mask_valid = children_flag.isin([0, 1])
    data = data.loc[mask_valid].copy()
    data["children_flag"] = children_flag[mask_valid]

    # Affair frequency is coded as an ordered numeric score (0, 1, 2, 3, 7, 12).
    affairs = data[affairs_col].astype(float)

    with_children = affairs[data["children_flag"] == 1]
    without_children = affairs[data["children_flag"] == 0]

    print("N with children:", len(with_children))
    print("N without children:", len(without_children))
    print("Mean affair score with children:", with_children.mean())
    print("Mean affair score without children:", without_children.mean())

    # Two-sample t-test on the affair frequency score.
    t_stat, p_val = stats.ttest_ind(
        with_children, without_children, equal_var=False, nan_policy="omit"
    )
    print("t-statistic (with vs without children):", t_stat)
    print("p-value:", p_val)

    # Also compute a simple Pearson correlation between children_flag and affair score.
    corr, corr_p = stats.pearsonr(data["children_flag"], affairs)
    print("Correlation (children_flag vs affair score):", corr)
    print("Correlation p-value:", corr_p)

    # Binary indicator: any extramarital intercourse in the past year.
    any_affair = (affairs > 0).astype(int)
    data["any_affair"] = any_affair

    # Contingency table for descriptive rates.
    crosstab = pd.crosstab(data["children_flag"], data["any_affair"])
    print("Contingency table (children_flag x any_affair):")
    print(crosstab)
    print("Affair rate with children:", any_affair[data["children_flag"] == 1].mean())
    print(
        "Affair rate without children:",
        any_affair[data["children_flag"] == 0].mean(),
    )

    # Logistic regression of any_affair on children_flag.
    X = sm.add_constant(data["children_flag"])
    model = sm.Logit(data["any_affair"], X).fit(disp=False)
    print(model.summary())


if __name__ == "__main__":
    main()

