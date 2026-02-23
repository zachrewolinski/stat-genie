import pandas as pd
import numpy as np
import statsmodels.formula.api as smf


def main() -> None:
    # Load dataset
    df = pd.read_csv("affairs.csv")

    # In this dataset, metadata indicates that:
    # - Column "age" actually encodes frequency of extramarital affairs (0,1,2,3,7,12).
    # - Column "religiousness" is a yes/no factor indicating whether there are children.
    #
    # We therefore construct:
    #   has_affair: binary indicator of any extramarital affair.
    #   children_yes: 1 if there are children in the marriage, 0 otherwise.
    df = df.copy()
    df["has_affair"] = (df["age"] > 0).astype(int)
    df["children_yes"] = df["religiousness"].astype(str).str.lower().eq("yes").astype(int)

    # Basic descriptive statistics: proportion with any affair and mean frequency.
    group_stats = (
        df.groupby("children_yes")
        .agg(
            n=("has_affair", "size"),
            prop_any_affair=("has_affair", "mean"),
            mean_affair_freq=("age", "mean"),
        )
        .reset_index()
    )

    # Logistic regression: probability of any affair as a function of children,
    # controlling for other observed covariates.
    formula = (
        "has_affair ~ children_yes + C(gender) + education + occupation + "
        "children + rating + yearsmarried + rownames"
    )

    logit_model = smf.logit(formula, data=df).fit(disp=False)
    params = logit_model.params
    conf_int = logit_model.conf_int()
    pvalues = logit_model.pvalues

    # Extract key results for the children indicator
    coef_children = params["children_yes"]
    p_children = pvalues["children_yes"]
    ci_low, ci_high = conf_int.loc["children_yes"]
    odds_ratio = float(np.exp(coef_children))
    ci_or_low = float(np.exp(ci_low))
    ci_or_high = float(np.exp(ci_high))

    # Print a concise summary that can be inspected outside this script.
    print("=== Descriptive statistics by children (children_yes: 0=no, 1=yes) ===")
    print(group_stats.to_string(index=False))
    print()
    print("=== Logistic regression: has_affair ~ children + controls ===")
    print(f"Coefficient for children_yes: {coef_children:.4f}")
    print(f"Odds ratio for children_yes: {odds_ratio:.4f}")
    print(
        "95% CI for odds ratio: "
        f"[{ci_or_low:.4f}, {ci_or_high:.4f}]"
    )
    print(f"p-value for children_yes: {p_children:.4g}")


if __name__ == "__main__":
    main()

