import pandas as pd
import statsmodels.formula.api as smf


def main() -> None:
    df = pd.read_csv("boxes.csv")

    # Majority choice: 1 = chose majority option, 0 = other (minority or undemonstrated)
    df["majority_choice"] = (df["majority_first"] == 2).astype(int)

    print("Dataset shape:", df.shape)
    print(df["majority_choice"].value_counts(normalize=True).rename("prop").to_frame())

    # Logistic regression with age (developmental stage proxy)
    print("\nLogistic regression: majority_choice ~ age")
    model_age = smf.logit("majority_choice ~ age", data=df).fit(disp=False)
    print(model_age.summary())

    # Logistic regression with cultural site (y) as categorical predictor
    print("\nLogistic regression: majority_choice ~ C(y)  (site differences)")
    model_site = smf.logit("majority_choice ~ C(y)", data=df).fit(disp=False)
    print(model_site.summary())

    # Combined model with both age and site
    print("\nLogistic regression: majority_choice ~ age + C(y)")
    model_both = smf.logit("majority_choice ~ age + C(y)", data=df).fit(disp=False)
    print(model_both.summary())


if __name__ == "__main__":
    main()

