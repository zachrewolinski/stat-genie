import numpy as np
import pandas as pd
import statsmodels.api as sm


def main() -> None:
    df = pd.read_csv("affairs.csv")

    # Basic derived variables
    df["any_affair"] = (df["affairs"] > 0).astype(int)
    df["children_yes"] = (df["children"] == "yes").astype(int)

    # Descriptive comparison by children status
    group = df.groupby("children").agg(
        mean_affairs=("affairs", "mean"),
        median_affairs=("affairs", "median"),
        prop_any_affair=("any_affair", "mean"),
        count=("affairs", "size"),
    )
    print("Descriptive statistics by children status:")
    print(group)
    print()

    # Logistic regression for any affair ~ children + controls
    covariates = [
        "children_yes",
        "age",
        "yearsmarried",
        "religiousness",
        "education",
        "occupation",
        "rating",
    ]
    X = df[covariates]
    X = sm.add_constant(X)
    y = df["any_affair"]

    logit_model = sm.Logit(y, X)
    logit_result = logit_model.fit(disp=False)

    print("Logistic regression: any_affair ~ children + controls")
    print(logit_result.summary())
    print()

    # Compute odds ratio for children_yes
    params = logit_result.params
    conf_int = logit_result.conf_int()
    beta_children = float(params["children_yes"])
    or_children = float(np.exp(beta_children))
    ci_low, ci_high = conf_int.loc["children_yes"]
    or_ci_low = float(np.exp(ci_low))
    or_ci_high = float(np.exp(ci_high))

    print(
        f"Children coefficient (log-odds): {beta_children:.3f}, "
        f"OR={or_children:.3f}, 95% CI [{or_ci_low:.3f}, {or_ci_high:.3f}]"
    )


if __name__ == "__main__":
    main()
