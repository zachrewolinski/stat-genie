import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main() -> None:
    df = pd.read_csv("affairs.csv")

    # Binary outcome: any extramarital affair in past year
    df["has_affair"] = (df["affairs"] > 0).astype(int)

    print("=== Basic group summaries by children ===")
    group_summary = (
        df.groupby("children")[["has_affair", "affairs"]]
        .agg(["mean", "std", "count"])
        .round(3)
    )
    print(group_summary)
    print()

    # Simple logistic regression: has_affair ~ children
    print("=== Logistic regression: has_affair ~ children ===")
    model_simple = smf.logit("has_affair ~ C(children)", data=df).fit(disp=False)
    print(model_simple.summary())
    print()

    # Logistic regression with key covariates
    print("=== Logistic regression with controls ===")
    formula_controls = (
        "has_affair ~ C(children) + age + yearsmarried + "
        "religiousness + education + occupation + rating + C(gender)"
    )
    model_controls = smf.logit(formula_controls, data=df).fit(disp=False)
    print(model_controls.summary())
    print()

    # Predicted probabilities from controlled model for a "typical" person
    numeric_cols = ["age", "yearsmarried", "religiousness", "education", "occupation", "rating"]
    base_values = {col: df[col].mean() for col in numeric_cols}
    base_values["gender"] = df["gender"].mode()[0]

    print("=== Predicted probability of any affair (controlled model) ===")
    for children_status in ["no", "yes"]:
        row = base_values.copy()
        row["children"] = children_status
        row_df = pd.DataFrame([row])
        pred_prob = model_controls.predict(row_df)[0]
        print(f"children = {children_status:3s} -> predicted P(has_affair=1) = {pred_prob:.3f}")


if __name__ == "__main__":
    main()

