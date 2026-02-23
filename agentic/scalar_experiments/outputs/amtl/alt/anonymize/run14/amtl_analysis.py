import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main() -> None:
    # Load data
    df = pd.read_csv("amtl.csv")

    # Rename columns to meaningful names
    df = df.rename(
        columns={
            "feature1": "tooth_class",
            "feature2": "specimen_id",
            "feature3": "missing",
            "feature4": "sockets",
            "feature5": "age",
            "feature6": "age_uncertainty",
            "feature7": "sex",
            "feature8": "genus",
            "feature9": "region",
        }
    )

    # Ensure we have positive numbers of observable sockets
    df = df[df["sockets"] > 0].copy()

    # Expand to tooth-level data: each observable socket becomes one row with a
    # binary AMTL outcome (1 = tooth missing, 0 = present).
    df = df.reset_index(drop=True)
    df["missing"] = df["missing"].astype(int)
    df["sockets"] = df["sockets"].astype(int)

    repeats = df["sockets"]
    tooth_df = df.loc[df.index.repeat(repeats)].copy()
    tooth_df["tooth_index"] = tooth_df.groupby(level=0).cumcount()
    tooth_df["amtl"] = (tooth_df["tooth_index"] < tooth_df["missing"]).astype(int)

    # Binomial regression: logit of AMTL probability as a function of genus, age, sex, and tooth class.
    formula = "amtl ~ C(genus) + age + sex + C(tooth_class)"
    model = smf.glm(
        formula=formula,
        data=tooth_df,
        family=sm.families.Binomial(),
    )
    result = model.fit()

    print(result.summary())

    # Extract genus-related coefficients
    print("\nGenus-related coefficients (log-odds):")
    for name, coef, pval in zip(result.params.index, result.params.values, result.pvalues.values):
        if "C(genus)" in name:
            print(f"{name:25s}  coef={coef: .4f}  p={pval: .4g}")

    # Predicted probabilities for each genus at mean age/sex and most common tooth class
    mean_age = tooth_df["age"].mean()
    mean_sex = tooth_df["sex"].mean()
    ref_tooth = tooth_df["tooth_class"].mode()[0]

    unique_genera = sorted(tooth_df["genus"].unique())
    pred_df = pd.DataFrame(
        {
            "genus": unique_genera,
            "age": mean_age,
            "sex": mean_sex,
            "tooth_class": ref_tooth,
        }
    )

    preds = result.predict(pred_df)
    print(
        f"\nPredicted AMTL probability by genus at age={mean_age:.2f}, "
        f"sex={mean_sex:.2f}, tooth_class={ref_tooth}"
    )
    for genus, p in zip(unique_genera, preds):
        print(f"{genus:12s}: {p:.4f}")


if __name__ == "__main__":
    main()
