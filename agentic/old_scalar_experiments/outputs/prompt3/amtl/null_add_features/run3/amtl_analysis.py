import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main() -> None:
    df = pd.read_csv("amtl.csv")

    # Basic derived measure: proportion of missing teeth in the class
    df["amtl_prop"] = df["num_amtl"] / df["sockets"]

    print("N rows:", len(df))
    print("\nRaw AMTL proportion by genus:")
    print(df.groupby("genus")["amtl_prop"].mean())

    # Binomial regression with logit link:
    # outcome: proportion missing, trials: sockets
    formula = (
        "amtl_prop ~ C(genus, Treatment(reference='Homo sapiens'))"
        " + age + prob_male + C(tooth_class)"
    )
    model = smf.glm(
        formula=formula,
        data=df,
        family=sm.families.Binomial(),
        freq_weights=df["sockets"],
    )
    result = model.fit()

    print("\nModel coefficients (subset for genus terms):")
    params = result.params
    conf_int = result.conf_int()
    pvalues = result.pvalues
    for genus in ["Pan", "Papio", "Pongo"]:
        name = f"C(genus, Treatment(reference='Homo sapiens'))[T.{genus}]"
        if name in params.index:
            print(
                f"{name}: coef={params[name]:.3f}, "
                f"CI=({conf_int.loc[name, 0]:.3f}, {conf_int.loc[name, 1]:.3f}), "
                f"p={pvalues[name]:.3g}"
            )

    # Standardized predicted AMTL probabilities by genus
    genera = ["Homo sapiens", "Pan", "Papio", "Pongo"]
    pred_means = {}
    for genus in genera:
        df_cf = df.copy()
        df_cf["genus"] = genus
        pred = result.predict(df_cf)
        pred_means[genus] = float(pred.mean())

    print("\nStandardized predicted AMTL proportions by genus:")
    for genus in genera:
        print(f"{genus}: {pred_means[genus]:.4f}")

    human = pred_means["Homo sapiens"]
    non_humans = [pred_means[g] for g in genera if g != "Homo sapiens"]
    print(
        "\nMean difference (Homo sapiens - average non-human genera): "
        f"{human - np.mean(non_humans):.4f}"
    )
    for genus in ["Pan", "Papio", "Pongo"]:
        print(
            f"Difference (Homo sapiens - {genus}): "
            f"{human - pred_means[genus]:.4f}"
        )


if __name__ == "__main__":
    main()

