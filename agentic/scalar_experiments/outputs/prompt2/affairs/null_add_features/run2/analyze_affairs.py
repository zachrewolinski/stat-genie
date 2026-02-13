import pandas as pd
import statsmodels.api as sm
import numpy as np


def main() -> None:
    df = pd.read_csv("affairs.csv")

    # Binary indicator for having any extramarital affairs in the past year
    df["any_affair"] = (df["affairs"] > 0).astype(int)

    # Basic descriptive statistics by children status
    desc = (
        df.groupby("children")
        .agg(
            mean_affairs=("affairs", "mean"),
            prop_any_affair=("any_affair", "mean"),
            count=("affairs", "size"),
        )
        .sort_index()
    )

    print("Descriptive statistics by children status:")
    print(desc.to_string())

    # Prepare data for logistic regression of any_affair on children, controlling for covariates
    model_df = df.dropna(
        subset=[
            "any_affair",
            "children",
            "age",
            "yearsmarried",
            "religiousness",
            "education",
            "occupation",
            "rating",
        ]
    ).copy()

    # Encode children as binary: yes=1, no=0
    model_df["children_bin"] = (model_df["children"] == "yes").astype(int)

    X = model_df[
        [
            "children_bin",
            "age",
            "yearsmarried",
            "religiousness",
            "education",
            "occupation",
            "rating",
        ]
    ]
    X = sm.add_constant(X)
    y = model_df["any_affair"]

    logit_model = sm.Logit(y, X).fit(disp=False)

    print("\nLogistic regression of any_affair on children and controls:")
    print(logit_model.summary())

    # Effect size for children
    params = logit_model.params
    conf_int = logit_model.conf_int()

    or_children = float(np.exp(params["children_bin"]))
    or_children_ci_low = float(np.exp(conf_int.loc["children_bin", 0]))
    or_children_ci_high = float(np.exp(conf_int.loc["children_bin", 1]))
    p_value = float(logit_model.pvalues["children_bin"])

    print(
        "\nEffect of having children (children_bin = 1 vs 0): "
        f"OR = {or_children:.3f}, 95% CI [{or_children_ci_low:.3f}, "
        f"{or_children_ci_high:.3f}], p-value = {p_value:.4g}"
    )


if __name__ == "__main__":
    main()

