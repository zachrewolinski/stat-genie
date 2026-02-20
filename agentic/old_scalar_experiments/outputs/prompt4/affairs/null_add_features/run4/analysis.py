import pandas as pd
import statsmodels.api as sm


def main() -> None:
    # Load data
    df = pd.read_csv("affairs.csv")

    # Create binary outcome: any extramarital affairs in past year
    df["affair_any"] = (df["affairs"] > 0).astype(int)

    # Basic group summaries by children status
    group_mean = df.groupby("children")["affairs"].agg(["mean", "std", "count"])
    group_prop_any = df.groupby("children")["affair_any"].mean()

    print("Mean number of affairs by children status:")
    print(group_mean)
    print("\nProportion with any affairs by children status:")
    print(group_prop_any)

    # Prepare covariates for a logistic regression of affair_any on children and controls
    # Encode children and gender as binary indicators
    df["children_yes"] = (df["children"] == "yes").astype(int)
    df["gender_male"] = (df["gender"] == "male").astype(int)

    covariates = [
        "children_yes",
        "age",
        "yearsmarried",
        "religiousness",
        "education",
        "occupation",
        "rating",
        "gender_male",
    ]

    # Drop rows with missing values in the variables used
    model_df = df[["affair_any"] + covariates].dropna()

    X = model_df[covariates]
    X = sm.add_constant(X)
    y = model_df["affair_any"]

    logit_model = sm.Logit(y, X).fit(disp=False)

    print("\nLogistic regression of any affair on children and controls:")
    print(logit_model.summary())
    print("\nCoefficient for children_yes (having children):")
    print(f"  coef = {logit_model.params['children_yes']:.4f}")
    print(f"  p-value = {logit_model.pvalues['children_yes']:.4g}")


if __name__ == "__main__":
    main()

