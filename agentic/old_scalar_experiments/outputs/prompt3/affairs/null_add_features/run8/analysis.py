import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main() -> None:
    df = pd.read_csv("affairs.csv")

    # Basic sanity checks
    print("Columns:", list(df.columns))
    print("Number of rows:", len(df))

    # Create a binary indicator for any affair
    df["any_affair"] = (df["affairs"] > 0).astype(int)

    # Summaries by children status
    grouped = df.groupby("children").agg(
        mean_affairs=("affairs", "mean"),
        prop_any_affair=("any_affair", "mean"),
        count=("affairs", "size"),
    )
    print("\nAffair summaries by children status:")
    print(grouped)

    # Logistic regression: probability of any affair ~ children + controls
    formula = "any_affair ~ C(children) + age + yearsmarried + religiousness + education + occupation + rating"
    logit_model = smf.logit(formula=formula, data=df).fit(disp=False)
    print("\nLogistic regression results (any_affair ~ children + controls):")
    print(logit_model.summary())

    # Poisson regression for affair counts as a robustness check
    poisson_model = smf.glm(
        formula="affairs ~ C(children) + age + yearsmarried + religiousness + education + occupation + rating",
        data=df,
        family=sm.families.Poisson(),
    ).fit()
    print("\nPoisson regression results (affairs ~ children + controls):")
    print(poisson_model.summary())


if __name__ == "__main__":
    main()

