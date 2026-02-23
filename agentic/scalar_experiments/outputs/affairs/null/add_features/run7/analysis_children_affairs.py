import pandas as pd
import statsmodels.formula.api as smf


def main() -> None:
    # Load data
    df = pd.read_csv("affairs.csv")

    # Keep core variables relevant to the research question and common controls
    cols = [
        "affairs",
        "children",
        "gender",
        "age",
        "yearsmarried",
        "religiousness",
        "education",
        "occupation",
        "rating",
    ]
    df = df[cols].copy()

    # Binary outcome: any extramarital affair in past year
    df["has_affair"] = (df["affairs"] > 0).astype(int)

    # Descriptive statistics by children status
    group = (
        df.groupby("children")
        .agg(
            mean_affairs=("affairs", "mean"),
            prop_has_affair=("has_affair", "mean"),
            n=("has_affair", "size"),
        )
        .sort_index()
    )

    print("Descriptive statistics by children status:")
    print(group)
    print()

    # Simple logistic regression: affair (yes/no) on children
    model_simple = smf.logit("has_affair ~ C(children)", data=df).fit(disp=False)
    print("Logistic regression: has_affair ~ C(children)")
    print(model_simple.summary())
    print()

    # Adjusted logistic regression with standard controls
    formula_full = (
        "has_affair ~ C(children) + gender + age + yearsmarried + "
        "religiousness + education + C(occupation) + rating"
    )
    model_full = smf.logit(formula_full, data=df).fit(disp=False)
    print("Adjusted logistic regression with controls:")
    print(model_full.summary())


if __name__ == "__main__":
    main()

