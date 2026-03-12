import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main() -> None:
    # Load data
    df = pd.read_csv("affairs.csv")

    # Binary indicator for any extramarital affair
    df["had_affair"] = (df["affairs"] > 0).astype(int)

    # Basic descriptives by children status
    summary = df.groupby("children")["had_affair"].agg(["mean", "sum", "count"])
    print("Had affair by children status:")
    print(summary)

    # Logistic regression: had_affair ~ children (yes/no)
    # Treat "no" as reference via C(children)
    model_simple = smf.logit("had_affair ~ C(children)", data=df).fit(disp=False)
    print("\nLogit had_affair ~ C(children):")
    print(model_simple.summary())

    # Logistic regression controlling for key covariates
    # Use age, yearsmarried, religiousness, rating as covariates
    model_adj = smf.logit(
        "had_affair ~ C(children) + age + yearsmarried + religiousness + rating",
        data=df,
    ).fit(disp=False)
    print("\nAdjusted logit had_affair ~ C(children) + covariates:")
    print(model_adj.summary())

    # Extract odds ratio and CI for children[T.yes] from adjusted model
    params = model_adj.params
    conf = model_adj.conf_int()
    or_children = np.exp(params["C(children)[T.yes]"])
    ci_low, ci_high = np.exp(conf.loc["C(children)[T.yes]"])
    p_value = model_adj.pvalues["C(children)[T.yes]"]

    print("\nAdjusted effect of having children (yes vs no):")
    print(f"Odds ratio: {or_children:.3f}")
    print(f"95% CI: [{ci_low:.3f}, {ci_high:.3f}]")
    print(f"p-value: {p_value:.4g}")


if __name__ == "__main__":
    main()
