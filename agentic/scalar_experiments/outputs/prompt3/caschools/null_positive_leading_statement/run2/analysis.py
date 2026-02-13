import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main() -> None:
    # Load dataset
    df = pd.read_csv("caschools.csv")

    # Construct key variables
    # Student–teacher ratio: students per teacher (lower is better)
    df["str"] = df["students"] / df["teachers"]

    # Overall academic performance: average of reading and math scores
    df["score"] = df[["read", "math"]].mean(axis=1)

    cols = [
        "str",
        "score",
        "income",
        "english",
        "lunch",
        "calworks",
        "expenditure",
        "computer",
        "students",
        "grades",
    ]
    data = df[cols].dropna()

    print(f"N (complete cases): {len(data)}")
    print(
        "Student–teacher ratio (str): "
        f"mean={data['str'].mean():.2f}, std={data['str'].std():.2f}, "
        f"min={data['str'].min():.2f}, max={data['str'].max():.2f}"
    )
    print(
        "Academic score (score): "
        f"mean={data['score'].mean():.2f}, std={data['score'].std():.2f}, "
        f"min={data['score'].min():.2f}, max={data['score'].max():.2f}"
    )

    # Simple correlation
    corr = data["str"].corr(data["score"])
    print(f"\nPearson corr(str, score) = {corr:.4f}")

    # Simple OLS: score ~ str
    X_simple = sm.add_constant(data["str"])
    y = data["score"]
    model_simple = sm.OLS(y, X_simple).fit()
    print("\nSimple OLS: score ~ str")
    print(model_simple.summary())

    # Multiple regression with basic controls
    formula = (
        "score ~ str + income + english + lunch + "
        "calworks + expenditure + computer + students + C(grades)"
    )
    model_multi = smf.ols(formula, data=data).fit()
    print("\nMultiple OLS:", formula)
    print(model_multi.summary())

    # Key coefficients for interpretation
    simple_coef = model_simple.params["str"]
    simple_p = model_simple.pvalues["str"]
    multi_coef = model_multi.params["str"]
    multi_p = model_multi.pvalues["str"]

    print(
        f"\nSimple model: coef(str) = {simple_coef:.4f}, "
        f"p-value = {simple_p:.4g}, R^2 = {model_simple.rsquared:.4f}"
    )
    print(
        f"Multiple model: coef(str) = {multi_coef:.4f}, "
        f"p-value = {multi_p:.4g}, R^2 = {model_multi.rsquared:.4f}"
    )


if __name__ == "__main__":
    main()

