import pandas as pd
import statsmodels.formula.api as smf


def main() -> None:
    # Load data
    df = pd.read_csv("affairs.csv")

    # Binary outcome: any extramarital affair in past year
    df["affair_any"] = (df["affairs"] > 0).astype(int)

    # Descriptive comparison by children status
    group = (
        df.groupby("children")["affair_any"]
        .agg(["mean", "count"])
        .rename(columns={"mean": "prevalence"})
    )
    print("Affair prevalence by children status:")
    print(group)
    print()

    # Logistic regression: outcome any affair, predictor children (yes/no)
    model_children = smf.logit("affair_any ~ C(children)", data=df).fit(disp=False)
    print("Logistic regression: affair_any ~ C(children)")
    print(model_children.summary())
    print()

    # Logistic regression with standard controls from the classic Affairs dataset
    # (gender, yearsmarried, religiousness, education, occupation, rating)
    formula_controls = (
        "affair_any ~ C(children) + C(gender) + yearsmarried + "
        "religiousness + education + occupation + rating"
    )
    model_controls = smf.logit(formula_controls, data=df).fit(disp=False)
    print("Logistic regression with controls:")
    print(model_controls.summary())


if __name__ == "__main__":
    main()

