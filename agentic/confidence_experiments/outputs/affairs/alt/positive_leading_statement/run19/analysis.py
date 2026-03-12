import math

import pandas as pd
import statsmodels.formula.api as smf


def main() -> None:
    df = pd.read_csv("affairs.csv")

    # Binary outcome: any extramarital affairs in past year
    df["has_affair"] = (df["affairs"] > 0).astype(int)

    print("Basic counts of has_affair by children:")
    counts = df.groupby("children")["has_affair"].agg(["mean", "sum", "count"])
    counts["percent_with_affair"] = counts["mean"] * 100
    print(counts)
    print()

    # Logistic regression controlling for key covariates
    formula = (
        "has_affair ~ C(children) + age + yearsmarried + religiousness + "
        "education + C(occupation) + C(gender) + rating"
    )
    model = smf.logit(formula=formula, data=df)
    result = model.fit(disp=False)

    print(result.summary())
    print()

    # Extract the children effect
    param_name = "C(children)[T.yes]"
    if param_name in result.params:
        beta = result.params[param_name]
        pvalue = result.pvalues[param_name]
        odds_ratio = math.exp(beta)

        print("Effect of having children (yes vs no):")
        print(f"  Log-odds coefficient: {beta:.4f}")
        print(f"  Odds ratio: {odds_ratio:.3f}")
        print(f"  p-value: {pvalue:.4g}")

        # Predicted probabilities at average covariates
        mean_row = df[
            ["age", "yearsmarried", "religiousness", "education", "rating"]
        ].mean()
        base = {
            "children": "no",
            "gender": df["gender"].mode()[0],
            "occupation": df["occupation"].mode()[0],
        }
        base.update(mean_row.to_dict())

        import pandas as _pd

        base_no = _pd.DataFrame([base])
        base_yes = base_no.copy()
        base_yes["children"] = "yes"

        pred_no = float(result.predict(base_no)[0])
        pred_yes = float(result.predict(base_yes)[0])
        diff = pred_yes - pred_no

        print()
        print("Predicted probability of any affair at typical covariates:")
        print(f"  No children : {pred_no:.3f}")
        print(f"  With children: {pred_yes:.3f}")
        print(f"  Difference (children - no children): {diff:.3f}")


if __name__ == "__main__":
    main()
