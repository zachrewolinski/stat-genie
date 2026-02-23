import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main() -> None:
    df = pd.read_csv("affairs.csv")

    # Basic cleaning: ensure expected columns exist
    required_cols = [
        "affairs",
        "children",
        "age",
        "yearsmarried",
        "religiousness",
        "education",
        "occupation",
        "rating",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print(f"Missing columns: {missing}")
        print(df.head())
        return

    # Create binary outcome: any affair in past year
    df["any_affair"] = (df["affairs"] > 0).astype(int)

    # Descriptive stats by children
    desc = (
        df.groupby("children")
        .agg(
            mean_affairs=("affairs", "mean"),
            std_affairs=("affairs", "std"),
            prop_any=("any_affair", "mean"),
            n=("affairs", "size"),
        )
        .reset_index()
    )

    print("Descriptive stats by children:")
    print(desc.to_string(index=False))

    # Difference in means test (Welch t-test) for number of affairs
    from scipy import stats

    affairs_yes = df.loc[df["children"] == "yes", "affairs"]
    affairs_no = df.loc[df["children"] == "no", "affairs"]
    t_stat, p_val_t = stats.ttest_ind(affairs_yes, affairs_no, equal_var=False)

    print("\nWelch t-test (affairs ~ children):")
    print(f"t = {t_stat:.3f}, p = {p_val_t:.4g}")

    # Difference in proportions for any affair
    any_yes = df.loc[df["children"] == "yes", "any_affair"]
    any_no = df.loc[df["children"] == "no", "any_affair"]
    n_yes = any_yes.shape[0]
    n_no = any_no.shape[0]
    p_yes = any_yes.mean()
    p_no = any_no.mean()

    # Pooled proportion z-test
    p_pool = (any_yes.sum() + any_no.sum()) / (n_yes + n_no)
    se = np.sqrt(p_pool * (1 - p_pool) * (1 / n_yes + 1 / n_no))
    z_stat = (p_yes - p_no) / se
    p_val_z = 2 * (1 - stats.norm.cdf(abs(z_stat)))

    print("\nProportion test (any_affair ~ children):")
    print(f"p_yes = {p_yes:.3f}, p_no = {p_no:.3f}")
    print(f"z = {z_stat:.3f}, p = {p_val_z:.4g}")

    # Logistic regression for any_affair controlling for covariates
    formula = "any_affair ~ C(children) + age + yearsmarried + religiousness + education + occupation + rating"
    logit_model = smf.logit(formula=formula, data=df).fit(disp=False)

    print("\nLogistic regression results (any_affair):")
    print(logit_model.summary())

    # Extract coefficient for children (yes vs no)
    # Depending on encoding, the term name may vary; print params for inspection
    print("\nLogit coefficients:")
    print(logit_model.params)


if __name__ == "__main__":
    main()

