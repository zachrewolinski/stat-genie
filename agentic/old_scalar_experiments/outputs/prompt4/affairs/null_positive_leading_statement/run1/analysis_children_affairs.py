import pandas as pd
import statsmodels.formula.api as smf


def main() -> None:
    df = pd.read_csv("affairs.csv")

    # Basic binary indicator: had at least one extramarital affair in past year
    df["had_affair"] = (df["affairs"] > 0).astype(int)

    # Descriptive statistics by children status
    group = df.groupby("children")
    desc = group[["affairs", "had_affair"]].agg(
        count=("affairs", "size"),
        mean_affairs=("affairs", "mean"),
        prop_any_affair=("had_affair", "mean"),
    )

    print("Descriptive stats by children status:")
    print(desc)
    print()

    # Logistic regression for having any affair, controlling for covariates
    # Treat children and gender as categorical.
    formula = (
        "had_affair ~ C(children) + C(gender) + age + yearsmarried + "
        "religiousness + education + occupation + rating"
    )

    logit_model = smf.logit(formula=formula, data=df)
    logit_res = logit_model.fit(disp=False)

    print("Logistic regression results (had_affair as outcome):")
    print(logit_res.summary())

    # Extract key effect for children
    params = logit_res.params
    conf = logit_res.conf_int()

    # children is coded as 'no' baseline, statsmodels will create C(children)[T.yes]
    coef_children = params.get("C(children)[T.yes]")
    ci_children = conf.loc["C(children)[T.yes]"].tolist()

    print()
    print("Effect of having children (log-odds, had_affair outcome):")
    print(f"coef = {coef_children:.4f}, 95% CI = [{ci_children[0]:.4f}, {ci_children[1]:.4f}]")


if __name__ == "__main__":
    main()

