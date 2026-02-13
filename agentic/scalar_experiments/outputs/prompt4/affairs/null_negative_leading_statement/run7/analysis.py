import pandas as pd
import statsmodels.formula.api as smf


def main() -> None:
    df = pd.read_csv("affairs.csv")

    # Binary outcome: any affairs in the past year
    df["any_affair"] = (df["affairs"] > 0).astype(int)

    # children is already yes/no; encode for modeling
    df["children_yes"] = (df["children"] == "yes").astype(int)

    # Logistic regression of any affair on children, controlling for key covariates
    model = smf.logit(
        "any_affair ~ children_yes + age + yearsmarried + religiousness + "
        "education + C(gender) + occupation + rating",
        data=df,
    ).fit(disp=False)

    # Average predicted probability of an affair for couples with vs without children
    df_with = df.copy()
    df_with["children_yes"] = 1
    df_without = df.copy()
    df_without["children_yes"] = 0

    pred_with = model.predict(df_with).mean()
    pred_without = model.predict(df_without).mean()

    diff = pred_with - pred_without

    print("Predicted affair rate with children:", float(pred_with))
    print("Predicted affair rate without children:", float(pred_without))
    print("Difference (with - without):", float(diff))
    print("Children coefficient:", float(model.params["children_yes"]))
    print("Children p-value:", float(model.pvalues["children_yes"]))


if __name__ == "__main__":
    main()

