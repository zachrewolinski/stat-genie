import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def main() -> None:
    df = pd.read_csv("affairs.csv")
    df = df.dropna().copy()
    df["any_affair"] = (df["affairs"] > 0).astype(int)

    group_summary = df.groupby("children").agg(
        mean_affairs=("affairs", "mean"),
        median_affairs=("affairs", "median"),
        prop_any_affair=("any_affair", "mean"),
        count=("affairs", "size"),
    )
    print("Group summary by children:")
    print(group_summary)

    formula = (
        "any_affair ~ C(children) + C(gender) + age + yearsmarried + "
        "religiousness + education + occupation + rating"
    )
    model = smf.logit(formula, data=df).fit(disp=False)

    print("\nLogistic regression results:")
    print(model.summary())

    params = model.params
    conf = model.conf_int()

    if "C(children)[T.yes]" in params.index:
        beta = params["C(children)[T.yes]"]
        pval = model.pvalues["C(children)[T.yes]"]
        or_value = float(np.exp(beta))
        ci_low = float(np.exp(conf.loc["C(children)[T.yes]", 0]))
        ci_high = float(np.exp(conf.loc["C(children)[T.yes]", 1]))
        print("\nChildren effect (yes vs no):")
        print(f"  Coefficient: {beta:.4f}")
        print(f"  Odds ratio: {or_value:.4f}")
        print(f"  95% CI for OR: ({ci_low:.4f}, {ci_high:.4f})")
        print(f"  p-value: {pval:.4g}")
    else:
        print("\nChildren effect term not found in model parameters.")


if __name__ == "__main__":
    main()

