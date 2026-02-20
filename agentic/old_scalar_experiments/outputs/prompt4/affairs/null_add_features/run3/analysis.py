import pandas as pd
import statsmodels.api as sm
import numpy as np


def main() -> None:
    # Load data
    df = pd.read_csv("affairs.csv")

    # Basic sanity: keep only rows with non-missing affairs and children
    df = df.dropna(subset=["affairs", "children"])

    # Binary outcome: any extramarital affair in past year
    df["any_affair"] = (df["affairs"] > 0).astype(int)

    # Encode children: 1 = yes, 0 = no
    df["children_yes"] = (df["children"].str.lower() == "yes").astype(int)

    # Descriptive statistics
    summary = []
    for has_children, sub in df.groupby("children_yes"):
        label = "yes" if has_children == 1 else "no"
        n = len(sub)
        prop_any = sub["any_affair"].mean()
        mean_affairs = sub["affairs"].mean()
        summary.append((label, n, prop_any, mean_affairs))

    print("Descriptive statistics by children status:")
    for label, n, prop_any, mean_affairs in summary:
        print(
            f"children={label:3s} | n={n:3d} | "
            f"Pr(any affair)={prop_any:0.3f} | mean(affairs)={mean_affairs:0.3f}"
        )

    # Logistic regression: any_affair ~ children_yes
    X = sm.add_constant(df["children_yes"])
    y = df["any_affair"]
    logit_model = sm.Logit(y, X)
    logit_res = logit_model.fit(disp=False)

    print("\nLogistic regression: any_affair ~ children_yes")
    print(logit_res.summary())

    # Odds ratio for having children vs not
    coef_children = logit_res.params["children_yes"]
    or_children = float(np.exp(coef_children))
    print(f"\nOdds ratio (children_yes vs no): {or_children:0.3f}")

    # Logistic regression with basic controls
    covariates = [
        "children_yes",
        "age",
        "yearsmarried",
        "religiousness",
        "education",
        "occupation",
        "rating",
    ]
    # Drop rows with missing covariates
    df_controls = df.dropna(subset=covariates)

    Xc = sm.add_constant(df_controls[covariates])
    yc = df_controls["any_affair"]
    logit_model_c = sm.Logit(yc, Xc)
    logit_res_c = logit_model_c.fit(disp=False)

    print(
        "\nLogistic regression with controls: "
        "any_affair ~ children_yes + age + yearsmarried + religiousness "
        "+ education + occupation + rating"
    )
    print(logit_res_c.summary())

    coef_children_c = logit_res_c.params["children_yes"]
    or_children_c = float(np.exp(coef_children_c))
    print(f"\nAdjusted odds ratio (children_yes vs no): {or_children_c:0.3f}")


if __name__ == "__main__":
    main()
