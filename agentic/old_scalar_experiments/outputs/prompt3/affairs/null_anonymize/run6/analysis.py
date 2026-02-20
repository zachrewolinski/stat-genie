import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def main() -> None:
    df = pd.read_csv("affairs.csv")

    # Binary indicator for any extramarital affairs in the past year
    df["affair_any"] = (df["feature2"] > 0).astype(int)

    # Indicator for having children in the marriage: 1 = yes, 0 = no
    df["children"] = (df["feature6"] == "yes").astype(int)

    # Basic group summaries
    group_affair_rate = df.groupby("children")["affair_any"].mean()
    group_affair_freq = df.groupby("children")["feature2"].mean()

    print("Prevalence of any affair by children (0=no, 1=yes):")
    print(group_affair_rate)
    print("\nMean affair frequency score by children (0=no, 1=yes):")
    print(group_affair_freq)

    # Logistic regression for any affair, controlling for observed covariates
    formula = (
        "affair_any ~ children + C(feature3) + feature4 + feature5 + "
        "feature7 + feature8 + feature9 + feature10"
    )
    logit_model = smf.logit(formula, data=df).fit(disp=False)

    print("\nLogistic regression results for any extramarital affair:")
    print(logit_model.summary())

    children_coef = float(logit_model.params["children"])
    children_pvalue = float(logit_model.pvalues["children"])
    children_or = float(np.exp(children_coef))

    print("\nChildren coefficient (log-odds):", children_coef)
    print("Children odds ratio:", children_or)
    print("Children p-value:", children_pvalue)


if __name__ == "__main__":
    main()
