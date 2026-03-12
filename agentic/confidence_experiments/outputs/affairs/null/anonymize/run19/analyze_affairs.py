import numpy as np
import pandas as pd
import statsmodels.api as sm


def main() -> None:
    df = pd.read_csv("affairs.csv")

    # Outcome variables
    df["any_affair"] = (df["feature2"] > 0).astype(int)

    # Recode children indicator: 1 = has children, 0 = no children
    df["children"] = (df["feature6"] == "yes").astype(int)

    # Basic descriptive statistics by children status
    group_stats = (
        df.groupby("children")
        .agg(
            mean_affair_freq=("feature2", "mean"),
            prop_any_affair=("any_affair", "mean"),
            n=("any_affair", "size"),
        )
    )
    print("Descriptive stats by children (1 = has children, 0 = no children):")
    print(group_stats)
    print()

    # Logistic regression: any affair ~ children + controls
    # Controls: gender (feature3), age (feature4), years married (feature5),
    # religiousness (feature7), education (feature8), occupation (feature9),
    # marriage rating (feature10).
    df_model = df.copy()

    # Create dummy variable for gender: 1 = male, 0 = female
    df_model["male"] = (df_model["feature3"] == "male").astype(int)

    X = df_model[
        [
            "children",
            "male",
            "feature4",  # age
            "feature5",  # years married
            "feature7",  # religiousness
            "feature8",  # education
            "feature9",  # occupation
            "feature10",  # marriage rating
        ]
    ]
    X = sm.add_constant(X, has_constant="add")
    y = df_model["any_affair"]

    logit_model = sm.Logit(y, X)
    result = logit_model.fit(disp=False)

    print("Logistic regression results: any_affair ~ children + controls")
    print(result.summary())
    print()

    # Odds ratio for children and its confidence interval
    params = result.params
    conf = result.conf_int()

    child_coef = float(params["children"])
    child_ci_low, child_ci_high = [float(v) for v in conf.loc["children"]]

    odds_ratio = float(np.exp(child_coef))
    ci_low_or = float(np.exp(child_ci_low))
    ci_high_or = float(np.exp(child_ci_high))

    print("Children coefficient (log-odds):", child_coef)
    print("95% CI (log-odds):", (child_ci_low, child_ci_high))
    print("Odds ratio for having children:", odds_ratio)
    print("95% CI for odds ratio:", (ci_low_or, ci_high_or))
    print("p-value for children:", result.pvalues["children"])


if __name__ == "__main__":
    main()
