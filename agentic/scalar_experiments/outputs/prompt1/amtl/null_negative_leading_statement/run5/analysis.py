import pandas as pd
import patsy
import statsmodels.api as sm


def main() -> None:
    df = pd.read_csv("amtl.csv")

    # Ensure expected columns are present
    required_cols = [
        "num_amtl",
        "sockets",
        "age",
        "prob_male",
        "genus",
        "tooth_class",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Drop rows with missing key variables and ensure positive socket counts
    df = df.dropna(subset=["num_amtl", "sockets", "age", "prob_male"]).copy()
    df = df[df["sockets"] > 0].copy()
    # Remove logically inconsistent rows where the number of missing teeth
    # exceeds the number of observable sockets for that tooth class.
    df = df[df["num_amtl"] <= df["sockets"]].copy()

    # Use a treatment-coded factor for genus with Homo sapiens as the reference
    # so other genus coefficients are interpreted relative to humans.
    exog_formula = (
        "C(genus, Treatment(reference='Homo sapiens')) "
        "+ age + prob_male + C(tooth_class)"
    )

    # Construct aggregated binomial response as [successes, failures]
    successes = df["num_amtl"]
    failures = df["sockets"] - df["num_amtl"]
    endog = pd.concat([successes, failures], axis=1).to_numpy()

    # Build design matrix for predictors
    X = patsy.dmatrix(exog_formula, df, return_type="dataframe")

    model = sm.GLM(endog, X, family=sm.families.Binomial())
    result = model.fit()

    print("=== Binomial regression summary ===")
    print(result.summary())

    # Compute marginal predicted AMTL probabilities for each genus, adjusted
    # for the observed distributions of age, sex, and tooth class.
    covariates = df[["genus", "age", "prob_male", "tooth_class"]].copy()
    genera = sorted(df["genus"].unique())

    design_info = X.design_info

    print("\n=== Marginal predicted AMTL probabilities by genus ===")
    for g in genera:
        pred_data = covariates.copy()
        pred_data["genus"] = g
        X_pred = patsy.build_design_matrices(
            [design_info], pred_data, return_type="dataframe"
        )[0]
        probs = result.predict(X_pred)
        print(f"{g}: mean={probs.mean():.4f}")


if __name__ == "__main__":
    main()
