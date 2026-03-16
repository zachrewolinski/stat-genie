import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from patsy.contrasts import Treatment


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def expand_to_sockets(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in df.iterrows():
        sockets = int(r["sockets"])
        if sockets <= 0:
            continue
        num_amtl = int(r["num_amtl"])
        if num_amtl < 0:
            num_amtl = 0
        if num_amtl > sockets:
            num_amtl = sockets
        for i in range(sockets):
            rows.append(
                {
                    "amtl": 1 if i < num_amtl else 0,
                    "genus": r["genus"],
                    "tooth_class": r["tooth_class"],
                    "age": float(r["age"]),
                    "prob_male": float(r["prob_male"]),
                }
            )
    return pd.DataFrame(rows)


def fit_model(sock_df: pd.DataFrame):
    formula = "amtl ~ C(genus, Treatment(reference='Homo sapiens')) + C(tooth_class) + age + prob_male"
    model = smf.glm(formula=formula, data=sock_df, family=sm.families.Binomial())
    result = model.fit()
    return result


def summarize_genus_effects(result, sock_df: pd.DataFrame):
    print(result.summary())

    print("\nGenus coefficients (vs Homo sapiens):")
    for name in result.params.index:
        if name.startswith("C(genus"):
            coef = result.params[name]
            pval = result.pvalues[name]
            print(f"{name}: coef={coef:.3f}, p-value={pval:.3g}")

    print("\nAverage predicted AMTL probability by genus (marginal over age/sex/tooth_class):")
    genera = sorted(sock_df["genus"].unique())
    base_df = sock_df.copy()
    for g in genera:
        new_df = base_df.copy()
        new_df["genus"] = g
        preds = result.predict(new_df)
        print(f"{g}: mean predicted P(AMTL) = {preds.mean():.4f}")


def main():
    df = load_data("amtl.csv")
    sock_df = expand_to_sockets(df)
    print(f"Original rows: {len(df)}, expanded socket rows: {len(sock_df)}")
    result = fit_model(sock_df)
    summarize_genus_effects(result, sock_df)


if __name__ == "__main__":
    main()

