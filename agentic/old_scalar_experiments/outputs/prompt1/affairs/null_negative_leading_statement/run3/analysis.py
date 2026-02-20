import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def main() -> None:
    df = pd.read_csv("affairs.csv")

    # Binary outcome: any extramarital affair in the past year
    df["had_affair"] = (df["affairs"] > 0).astype(int)

    # Basic group-level summaries by presence of children
    group = (
        df.groupby("children")["had_affair"]
        .agg(["mean", "sum", "count"])
        .rename(columns={"mean": "prop_with_affair", "sum": "num_with_affair"})
    )

    print("=== Grouped proportions of any affair by children ===")
    print(group)
    print()

    # Mean number of affairs (including zeros) by children
    mean_affairs = df.groupby("children")["affairs"].mean()
    print("=== Mean number of affairs (including zeros) by children ===")
    print(mean_affairs)
    print()

    # Logistic regression: had_affair ~ children + controls
    # Controls: age, yearsmarried, religiousness, education, occupation, rating, gender
    formula = (
        "had_affair ~ C(children) + age + yearsmarried + "
        "religiousness + education + occupation + rating + C(gender)"
    )

    model = smf.logit(formula=formula, data=df).fit(disp=False)

    print("=== Logistic regression results ===")
    print(model.summary())
    print()

    # Odds ratios and p-values for easier interpretation
    params = model.params
    conf = model.conf_int()
    conf.columns = ["ci_lower", "ci_upper"]
    or_table = np.exp(pd.concat([params, conf], axis=1))
    or_table.columns = ["odds_ratio", "ci_lower", "ci_upper"]

    print("=== Odds ratios (exp(beta)) with 95% CI ===")
    print(or_table)
    print()

    print("=== P-values ===")
    print(model.pvalues)


if __name__ == "__main__":
    main()

