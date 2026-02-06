import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main():
    df = pd.read_csv("affairs.csv")

    # Basic descriptive stats
    df["has_affair"] = (df["affairs"] > 0).astype(int)

    # Group means by children status
    group_stats = df.groupby("children").agg(
        mean_affairs=("affairs", "mean"),
        median_affairs=("affairs", "median"),
        affair_rate=("has_affair", "mean"),
        count=("affairs", "size"),
    )

    # Encode children as binary for regression (yes=1, no=0)
    df["children_yes"] = (df["children"] == "yes").astype(int)

    # OLS on affairs count (simple)
    ols_simple = smf.ols("affairs ~ children_yes", data=df).fit()

    # OLS with controls
    ols_controls = smf.ols(
        "affairs ~ children_yes + age + yearsmarried + religiousness + education + occupation + rating + C(gender)",
        data=df,
    ).fit()

    # Logistic regression on any affair
    logit_simple = smf.logit("has_affair ~ children_yes", data=df).fit(disp=False)
    logit_controls = smf.logit(
        "has_affair ~ children_yes + age + yearsmarried + religiousness + education + occupation + rating + C(gender)",
        data=df,
    ).fit(disp=False)

    print("Group stats by children:\n", group_stats, "\n")

    def summarize(model, name):
        coef = model.params["children_yes"]
        pval = model.pvalues["children_yes"]
        return f"{name}: coef(children_yes)={coef:.4f}, p-value={pval:.4g}"

    print(summarize(ols_simple, "OLS simple"))
    print(summarize(ols_controls, "OLS controls"))
    print(summarize(logit_simple, "Logit simple"))
    print(summarize(logit_controls, "Logit controls"))


if __name__ == "__main__":
    main()
