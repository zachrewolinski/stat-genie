import numpy as np
import pandas as pd
import statsmodels.api as sm


def main():
    df = pd.read_csv("mortgage.csv")

    # Columns used in models
    control_cols = [
        "feature2",
        "feature3",
        "feature4",
        "feature5",
        "feature6",
        "feature7",
        "feature8",
        "feature9",
        "feature10",
        "feature12",
        "feature13",
        "feature14",
    ]

    df_model = df[control_cols].dropna()

    # Outcome: 1 if accepted, 0 if denied
    y = df_model["feature14"]

    # Basic rate comparison
    rates = df_model.groupby("feature2")["feature14"].mean().rename({0: "male", 1: "female"})
    counts = df_model["feature2"].value_counts().rename({0: "male", 1: "female"})

    # Simple model: approval ~ female
    X_simple = sm.add_constant(df_model[["feature2"]])
    model_simple = sm.Logit(y, X_simple).fit(disp=False)

    # Controlled model: approval ~ female + applicant/credit covariates
    control_cols_no_y = [c for c in control_cols if c != "feature14"]
    X_ctrl = sm.add_constant(df_model[control_cols_no_y])
    model_ctrl = sm.Logit(y, X_ctrl).fit(disp=False)

    # Extract key stats for gender coefficient
    coef_simple = float(model_simple.params["feature2"])
    p_simple = float(model_simple.pvalues["feature2"])
    or_simple = float(np.exp(coef_simple))

    coef_ctrl = float(model_ctrl.params["feature2"])
    p_ctrl = float(model_ctrl.pvalues["feature2"])
    or_ctrl = float(np.exp(coef_ctrl))

    # Print summary for downstream reading
    print(f"Rows used after dropna: {len(df_model)}")
    print("Approval rates by gender:")
    print(rates)
    print("Counts by gender:")
    print(counts)
    print("\nSimple logit (approval ~ female):")
    print(model_simple.summary())
    print("\nControlled logit (approval ~ female + covariates):")
    print(model_ctrl.summary())

    print("\nKey gender effect stats:")
    print(f"Simple model coef={coef_simple:.4f}, OR={or_simple:.4f}, p={p_simple:.4g}")
    print(f"Controlled model coef={coef_ctrl:.4f}, OR={or_ctrl:.4f}, p={p_ctrl:.4g}")


if __name__ == "__main__":
    main()
