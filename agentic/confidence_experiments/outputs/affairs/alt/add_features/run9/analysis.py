import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main() -> None:
    df = pd.read_csv("affairs.csv")

    # Binary outcome: any extramarital affair in the past year
    df["any_affair"] = (df["affairs"] > 0).astype(int)

    # Basic descriptive statistics by children
    desc = (
        df.groupby("children")
        .agg(
            mean_affairs=("affairs", "mean"),
            std_affairs=("affairs", "std"),
            any_affair_rate=("any_affair", "mean"),
            n=("affairs", "size"),
        )
        .reset_index()
    )
    print("Descriptive statistics by children:")
    print(desc.to_string(index=False))
    print()

    # Unadjusted logistic regression: any_affair ~ children
    print("Unadjusted logistic regression: any_affair ~ C(children)")
    model_unadj = smf.logit("any_affair ~ C(children)", data=df).fit(disp=False)
    print(model_unadj.summary())
    print()

    # Compute odds ratio for having children (yes vs no)
    params_unadj = model_unadj.params
    if "C(children)[T.yes]" in params_unadj:
        or_children = float(np.exp(params_unadj["C(children)[T.yes]"]))
        print(f"Unadjusted odds ratio for children (yes vs no): {or_children:.3f}")
    print()

    # Adjusted logistic regression with key demographic and marital variables.
    # This mirrors common analyses of the Fair affairs dataset.
    formula_adj = (
        "any_affair ~ C(children) + C(gender) + age + yearsmarried + "
        "religiousness + education + occupation + rating"
    )
    print("Adjusted logistic regression:")
    print(formula_adj)
    model_adj = smf.logit(formula_adj, data=df).fit(disp=False)
    print(model_adj.summary())
    print()

    # Odds ratio and predicted probabilities from adjusted model
    params_adj = model_adj.params
    if "C(children)[T.yes]" in params_adj:
        or_children_adj = float(np.exp(params_adj["C(children)[T.yes]"]))
        print(
            f"Adjusted odds ratio for children (yes vs no): "
            f"{or_children_adj:.3f}"
        )

    # Predicted probabilities for a \"typical\" profile with and without children
    # (using sample means for numeric covariates and the most frequent category
    #  for categorical covariates).
    typical = {}
    for col in ["age", "yearsmarried", "religiousness", "education", "occupation", "rating"]:
        typical[col] = df[col].mean()
    for col in ["gender"]:
        typical[col] = df[col].mode().iat[0]

    for children_status in ["no", "yes"]:
        typical_row = typical.copy()
        typical_row["children"] = children_status
        prob = model_adj.predict(pd.DataFrame([typical_row]))[0]
        print(
            f"Predicted probability of any affair for children={children_status}: "
            f"{prob:.3f}"
        )


if __name__ == "__main__":
    main()
