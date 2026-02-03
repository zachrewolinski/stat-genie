import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main() -> None:
    df = pd.read_csv("panda_nuts.csv")

    # Efficiency as rate (nuts opened per second)
    df["rate"] = df["nuts_opened"] / df["seconds"]

    # Count model with exposure (seconds) -> models efficiency directly
    poisson_model = smf.glm(
        "nuts_opened ~ age + C(sex) + C(help)",
        data=df,
        family=sm.families.Poisson(),
        offset=np.log(df["seconds"]),
    )
    poisson_res = poisson_model.fit()

    # OLS on rate for a simple, interpretable check (robust SE)
    ols_res = smf.ols(
        "rate ~ age + C(sex) + C(help)",
        data=df,
    ).fit(cov_type="HC3")

    print("=== Poisson model (with log(seconds) offset) ===")
    print(poisson_res.summary())
    print()
    print("Wald test (age, sex, help jointly):")
    print(poisson_res.wald_test("age = 0, C(sex)[T.m] = 0, C(help)[T.y] = 0"))
    print()

    print("=== OLS on rate (nuts_opened / seconds), robust SE ===")
    print(ols_res.summary())
    print()
    print("F-test (age, sex, help jointly):")
    print(ols_res.f_test("age = 0, C(sex)[T.m] = 0, C(help)[T.y] = 0"))


if __name__ == "__main__":
    main()
