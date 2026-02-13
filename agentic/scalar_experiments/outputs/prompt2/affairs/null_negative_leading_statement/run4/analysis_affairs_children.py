import pandas as pd
import statsmodels.formula.api as smf


def main() -> None:
    # Load data
    df = pd.read_csv("affairs.csv")

    # Create a binary indicator for having any extramarital affairs in the past year
    df["any_affair"] = (df["affairs"] > 0).astype(int)

    # Basic descriptive statistics by children status
    grouped = df.groupby("children")["any_affair"].agg(["mean", "sum", "count"])
    print("Affair incidence by children status:")
    print(grouped)
    print()

    grouped_affairs = df.groupby("children")["affairs"].agg(["mean", "median"])
    print("Number of affairs (coded scale) by children status:")
    print(grouped_affairs)
    print()

    # Logistic regression: probability of any affair as a function of children and controls
    # children is a yes/no factor; other covariates follow the Fair (1978) specification
    formula = (
        "any_affair ~ C(children) + C(gender) + age + yearsmarried + "
        "religiousness + education + occupation + rating"
    )
    model = smf.logit(formula=formula, data=df)
    result = model.fit(disp=False)

    print("Logistic regression results (any_affair ~ children + controls):")
    print(result.summary())
    print()

    # Extract the coefficient for children (yes vs no) if present
    children_coef = None
    children_pvalue = None
    for param_name, coef in result.params.items():
        if "C(children)" in param_name:
            children_coef = coef
            children_pvalue = result.pvalues[param_name]
            break

    print("Children effect (log-odds) and p-value from the model:")
    print(f"coef={children_coef}, p-value={children_pvalue}")


if __name__ == "__main__":
    main()

