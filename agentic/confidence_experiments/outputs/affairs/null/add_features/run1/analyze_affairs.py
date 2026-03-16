import pandas as pd
import numpy as np
import statsmodels.formula.api as smf


def main() -> None:
    df = pd.read_csv("affairs.csv")

    # Create a binary outcome: any extramarital affair in the past year.
    df["any_affair"] = (df["affairs"] > 0).astype(int)

    # Basic group-wise summaries by children status.
    summary = (
        df.groupby("children")
        .agg(
            n=("any_affair", "size"),
            mean_affairs=("affairs", "mean"),
            prop_any_affair=("any_affair", "mean"),
        )
        .reset_index()
    )

    print("Group-wise summary by children status:")
    print(summary.to_string(index=False))
    print("\n")

    # Unadjusted logistic regression: any_affair ~ children
    model_unadj = smf.logit("any_affair ~ C(children)", data=df).fit(disp=False)
    print("Unadjusted logistic regression (any_affair ~ C(children)):")
    print(model_unadj.summary())
    print("\n")

    # Adjusted logistic regression including key demographic/marital covariates
    # present in the data.
    covariates = ["age", "yearsmarried", "religiousness", "education", "occupation", "rating"]
    # Only keep rows without missing values on these covariates.
    df_adj = df.dropna(subset=["any_affair", "children"] + covariates)

    model_adj = smf.logit(
        "any_affair ~ C(children) + age + yearsmarried + religiousness + education + occupation + rating",
        data=df_adj,
    ).fit(disp=False)

    print("Adjusted logistic regression (any_affair ~ C(children) + covariates):")
    print(model_adj.summary())
    print("\n")

    # Extract the children effect from both models.
    for name, model in [("unadjusted", model_unadj), ("adjusted", model_adj)]:
        params = model.params
        bse = model.bse
        # C(children)[T.yes] is the log-odds difference for 'yes' vs baseline ('no').
        key = "C(children)[T.yes]"
        if key in params:
            coef = params[key]
            se = bse[key]
            z = coef / se
            p_value = model.pvalues[key]
            or_est = float(np.exp(coef))
            ci_low = float(np.exp(coef - 1.96 * se))
            ci_high = float(np.exp(coef + 1.96 * se))
            print(f"{name.capitalize()} model effect of children (yes vs no):")
            print(f"  log-odds coef = {coef:.3f}, SE = {se:.3f}, z = {z:.3f}, p = {p_value:.4f}")
            print(f"  odds ratio = {or_est:.3f} (95% CI: {ci_low:.3f}, {ci_high:.3f})")
        else:
            print(f"{name.capitalize()} model did not contain coefficient for C(children)[T.yes].")
        print()


if __name__ == "__main__":
    main()

