import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main() -> None:
    # Load dataset
    df = pd.read_csv("amtl.csv")

    # Rename columns for clarity
    df = df.rename(
        columns={
            "feature1": "tooth_class",
            "feature2": "specimen_id",
            "feature3": "missing_teeth",
            "feature4": "observable_sockets",
            "feature5": "age",
            "feature6": "age_uncertainty",
            "feature7": "sex_estimate",
            "feature8": "genus",
            "feature9": "region",
        }
    )

    # Construct binomial response: missing vs present teeth
    df["total_teeth"] = df["missing_teeth"] + df["observable_sockets"]
    df = df[df["total_teeth"] > 0].copy()
    df["amtl_rate"] = df["missing_teeth"] / df["total_teeth"]

    # Drop any rows with amtl_rate outside [0, 1] just in case
    df = df[(df["amtl_rate"] >= 0) & (df["amtl_rate"] <= 1)].copy()

    # Binomial GLM: AMTL rate as a function of genus, age, sex, and tooth class
    # Use Pan (chimpanzees) as the reference genus.
    formula = (
        "amtl_rate ~ C(genus, Treatment(reference='Pan'))"
        " + age + sex_estimate + C(tooth_class)"
    )

    model = smf.glm(
        formula=formula,
        data=df,
        family=sm.families.Binomial(),
        freq_weights=df["total_teeth"],
    )
    result = model.fit()

    print(result.summary())

    # Predicted AMTL rates by genus at typical covariate values
    mean_age = df["age"].mean()
    mean_sex = df["sex_estimate"].mean()
    modal_tooth_class = df["tooth_class"].mode().iat[0]

    genera = sorted(df["genus"].unique())
    pred_df = pd.DataFrame(
        {
            "genus": genera,
            "age": mean_age,
            "sex_estimate": mean_sex,
            "tooth_class": modal_tooth_class,
        }
    )

    pred_df["predicted_amtl_rate"] = result.predict(pred_df)

    print("\nPredicted AMTL rate by genus at mean covariates:")
    print(pred_df)

    # Extract Homo sapiens vs Pan contrast
    param_name = "C(genus, Treatment(reference='Pan'))[T.Homo sapiens]"
    if param_name in result.params.index:
        coef = result.params[param_name]
        p_value = result.pvalues[param_name]
        print("\nEffect of Homo sapiens vs Pan (log-odds):")
        print(f"coef = {coef:.3f}, p-value = {p_value:.3g}")
    else:
        print(
            "\nCould not find Homo sapiens vs Pan coefficient; "
            "check model parameterization."
        )


if __name__ == "__main__":
    main()

