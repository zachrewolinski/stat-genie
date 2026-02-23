import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main() -> None:
    df = pd.read_csv("affairs.csv")

    df["any_affair"] = (df["affairs"] > 0).astype(int)

    grouped = (
        df.groupby("children")
        .agg(
            mean_affairs=("affairs", "mean"),
            median_affairs=("affairs", "median"),
            prop_with_affair=("any_affair", "mean"),
            n=("any_affair", "size"),
        )
        .reset_index()
    )

    print("Descriptive statistics by children status:")
    print(grouped.to_string(index=False))
    print("\nLogistic regression: any_affair ~ children")

    model_simple = smf.logit("any_affair ~ C(children)", data=df).fit(disp=False)
    print(model_simple.summary())

    print("\nOdds ratios (simple model):")
    or_simple = model_simple.params.copy()
    or_simple = or_simple.apply(lambda x: x if pd.isna(x) else float(pd.np.exp(x)))
    print(or_simple)

    print("\nLogistic regression with controls:")
    formula_full = (
        "any_affair ~ C(children) + age + yearsmarried + religiousness + "
        "education + occupation + rating"
    )
    model_full = smf.logit(formula_full, data=df).fit(disp=False)
    print(model_full.summary())

    print("\nOdds ratios (full model):")
    or_full = model_full.params.copy()
    or_full = or_full.apply(lambda x: x if pd.isna(x) else float(pd.np.exp(x)))
    print(or_full)


if __name__ == "__main__":
    main()

