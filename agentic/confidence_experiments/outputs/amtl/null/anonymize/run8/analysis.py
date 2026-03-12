import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main() -> None:
    # Load dataset
    df = pd.read_csv("amtl.csv")
    print(f"Raw rows: {len(df)}")

    # Basic cleaning: keep rows with valid counts and key covariates
    df = df[df["feature4"] > 0].copy()
    df = df.dropna(
        subset=["feature3", "feature4", "feature5", "feature7", "feature1", "feature8"]
    )
    print(f"Rows after cleaning: {len(df)}")

    # Outcome: proportion of missing teeth per observable socket
    df["prop_missing"] = df["feature3"] / df["feature4"]

    print("\nUnique genera and counts:")
    print(df["feature8"].value_counts())
    print("\nTooth class counts:")
    print(df["feature1"].value_counts())

    # Determine human label (any genus starting with 'Homo')
    genera = df["feature8"].unique().tolist()
    human_candidates = [g for g in genera if isinstance(g, str) and g.startswith("Homo")]
    human_label = human_candidates[0] if human_candidates else None

    if human_label is not None:
        nonhuman = [g for g in genera if g != human_label]
        genus_order = [human_label] + sorted(nonhuman)
    else:
        genus_order = sorted(genera)

    # Set categorical encodings with human (if present) as baseline
    df["feature8"] = pd.Categorical(df["feature8"], categories=genus_order)
    df["feature1"] = pd.Categorical(df["feature1"])

    print("\nGenus order used in model:", genus_order)
    if human_label is not None:
        print(f"Human genus label detected: {human_label}")
    else:
        print("Warning: No genus starting with 'Homo' detected.")

    # Binomial regression: proportion missing with number of observable sockets as weights
    model = smf.glm(
        formula="prop_missing ~ C(feature8) + feature5 + feature7 + C(feature1)",
        data=df,
        family=sm.families.Binomial(),
        freq_weights=df["feature4"],
    )
    result = model.fit()

    print("\n=== GLM Binomial results ===")
    print(result.summary())

    # Extract genus coefficients and p-values
    label = human_label or genus_order[0]
    print(f"\nGenus effects relative to baseline genus '{label}':")
    for name, coef, pval in zip(
        result.params.index, result.params.values, result.pvalues.values
    ):
        if "C(feature8)" in name:
            print(f"{name}: coef={coef:.3f}, p={pval:.4g}")

    # Predicted probabilities at representative covariate values
    mean_age = float(df["feature5"].mean())
    mean_sex = float(df["feature7"].mean())
    modal_tooth_class = df["feature1"].mode()[0]
    print(
        "\nPredicted AMTL probability per socket at mean age "
        f"({mean_age:.2f}), mean sex estimate ({mean_sex:.2f}), "
        f"and tooth class '{modal_tooth_class}':"
    )

    for genus in genus_order:
        new = pd.DataFrame(
            {
                "feature5": [mean_age],
                "feature7": [mean_sex],
                "feature1": [modal_tooth_class],
                "feature8": [genus],
            }
        )
        new["feature8"] = pd.Categorical(new["feature8"], categories=df["feature8"].cat.categories)
        new["feature1"] = pd.Categorical(new["feature1"], categories=df["feature1"].cat.categories)
        pred = float(result.predict(new)[0])
        print(f"{genus}: {pred:.3f}")


if __name__ == "__main__":
    main()

