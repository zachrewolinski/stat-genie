import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main() -> None:
    df = pd.read_csv("affairs.csv")

    # According to the metadata, the `age` column actually encodes
    # frequency of extramarital intercourse in the past year.
    # Treat any non‑zero value as having had an affair.
    df["has_affair"] = (df["age"] > 0).astype(int)

    # The metadata states that `religiousness` is actually the
    # answer to "Are there children in the marriage?" (yes/no).
    df["children_yes"] = (df["religiousness"] == "yes").astype(int)

    # Basic descriptive stats: affair prevalence overall and by children status.
    overall_rate = df["has_affair"].mean()
    by_children = df.groupby("children_yes")["has_affair"].mean()

    # Also compare mean affair frequency (`age`) by children status.
    freq_by_children = df.groupby("children_yes")["age"].mean()

    # Simple logistic regression of having any affair on children status,
    # plus a few standard controls from the classic Fair affairs dataset.
    # Here we use the column names as given, even though the metadata
    # suggests some are mis‑labelled, and treat them as generic controls.
    df["gender_male"] = (df["gender"] == "male").astype(int)

    model = smf.logit(
        "has_affair ~ children_yes + yearsmarried + rating + gender_male",
        data=df,
    ).fit(disp=False)

    print("Overall affair prevalence (any affair):", overall_rate)
    print("Affair prevalence by children_yes (0=no, 1=yes):")
    print(by_children)
    print("\nMean affair frequency (age) by children_yes (0=no, 1=yes):")
    print(freq_by_children)
    print("\nLogit coefficients:")
    print(model.params)
    print("\nLogit summary:")
    print(model.summary())


if __name__ == "__main__":
    main()

