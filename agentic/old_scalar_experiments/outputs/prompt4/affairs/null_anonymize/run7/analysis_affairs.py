import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main() -> None:
    # Load data
    df = pd.read_csv("affairs.csv")

    # Rename columns to meaningful labels for clarity
    df = df.rename(
        columns={
            "feature1": "id_like",
            "feature2": "affair_freq",
            "feature3": "gender",
            "feature4": "age",
            "feature5": "years_married",
            "feature6": "children",
            "feature7": "religiousness",
            "feature8": "education",
            "feature9": "occupation",
            "feature10": "marriage_rating",
        }
    )

    # Binary indicator for any extramarital affair in past year
    df["affair_any"] = (df["affair_freq"] > 0).astype(int)

    # Basic descriptive stats by children status
    print("Descriptive statistics by children status")
    print(df.groupby("children")["affair_freq"].agg(["mean", "std", "count"]))
    print()
    print("Proportion with any affairs by children status")
    print(df.groupby("children")["affair_any"].mean())
    print()

    # Logistic regression: probability of any affair ~ children + controls
    # children is coded as factor ("yes"/"no"), so we use formula with C(children)
    formula = (
        "affair_any ~ C(children) + C(gender) + age + years_married "
        "+ religiousness + education + occupation + marriage_rating"
    )
    logit_model = smf.logit(formula=formula, data=df).fit(disp=False)

    print("Logistic regression results: affair_any ~ children + controls")
    print(logit_model.summary())
    print()

    # Also run a linear probability model for easier interpretation of marginal effect
    lpm = smf.ols(formula=formula, data=df).fit()
    print("Linear probability model (OLS) results")
    print(lpm.summary())


if __name__ == "__main__":
    main()

