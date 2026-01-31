import pandas as pd
import numpy as np
import statsmodels.api as sm
import patsy


def fit_model(df: pd.DataFrame):
    # Binomial GLM with successes/failures to model AMTL rate per socket
    endog = np.column_stack([df["num_amtl"], df["sockets"] - df["num_amtl"]])
    exog = patsy.dmatrix(
        "C(genus) + age + prob_male + C(tooth_class)",
        data=df,
        return_type="dataframe",
    )
    model = sm.GLM(endog, exog, family=sm.families.Binomial())
    return model.fit(), exog.design_info


def adjusted_mean_predictions(df: pd.DataFrame, result, design_info):
    means = {}
    for genus in ["Homo sapiens", "Pan", "Papio", "Pongo"]:
        df_g = df.copy()
        df_g["genus"] = genus
        exog_g = patsy.build_design_matrices([design_info], df_g)[0]
        pred = result.predict(exog_g)
        means[genus] = float(np.mean(pred))
    return means


def main():
    df = pd.read_csv("amtl.csv")

    result, design_info = fit_model(df)
    means = adjusted_mean_predictions(df, result, design_info)

    print(result.summary())
    print("\nAdjusted mean AMTL rate (predicted probability per socket):")
    for genus, mean in means.items():
        print(f"{genus}: {mean:.4f}")

    # Extract genus coefficients vs Homo sapiens baseline
    genus_terms = {
        "Pan": result.params.get("C(genus)[T.Pan]", np.nan),
        "Papio": result.params.get("C(genus)[T.Papio]", np.nan),
        "Pongo": result.params.get("C(genus)[T.Pongo]", np.nan),
    }
    genus_pvals = {
        "Pan": result.pvalues.get("C(genus)[T.Pan]", np.nan),
        "Papio": result.pvalues.get("C(genus)[T.Papio]", np.nan),
        "Pongo": result.pvalues.get("C(genus)[T.Pongo]", np.nan),
    }

    print("\nGenus log-odds differences vs Homo sapiens (positive => higher AMTL than Homo):")
    for genus in ["Pan", "Papio", "Pongo"]:
        coef = genus_terms[genus]
        pval = genus_pvals[genus]
        print(f"{genus}: coef={coef:.4f}, p={pval:.4g}")


if __name__ == "__main__":
    main()
