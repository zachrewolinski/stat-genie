import pandas as pd
import statsmodels.api as sm


def main() -> None:
    # Load data
    df = pd.read_csv("caschools.csv")

    # Construct key variables
    df["stratio"] = df["students"] / df["teachers"]
    df["avgscore"] = (df["read"] + df["math"]) / 2.0

    # Basic correlations
    corr_stratio_avg = df["stratio"].corr(df["avgscore"])
    corr_stratio_read = df["stratio"].corr(df["read"])
    corr_stratio_math = df["stratio"].corr(df["math"])

    print("Correlation student-teacher ratio vs avg score:", corr_stratio_avg)
    print("Correlation student-teacher ratio vs reading:", corr_stratio_read)
    print("Correlation student-teacher ratio vs math:", corr_stratio_math)

    # Simple linear regression of avgscore on stratio
    X = sm.add_constant(df["stratio"])
    model = sm.OLS(df["avgscore"], X).fit()
    print("\nOLS(avgscore ~ stratio)")
    print(model.summary())

    # Multiple regression controlling for key demographics and resources
    controls = [
        "income",
        "english",
        "calworks",
        "lunch",
        "computer",
        "expenditure",
    ]
    available_controls = [c for c in controls if c in df.columns]
    if available_controls:
        X_ctrl = sm.add_constant(df[["stratio"] + available_controls])
        model_ctrl = sm.OLS(df["avgscore"], X_ctrl).fit()
        print("\nOLS(avgscore ~ stratio + controls)")
        print("Controls used:", available_controls)
        print(model_ctrl.summary())



if __name__ == "__main__":
    main()
