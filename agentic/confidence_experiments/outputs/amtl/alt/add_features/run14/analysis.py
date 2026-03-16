import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def expand_to_teeth(df: pd.DataFrame) -> pd.DataFrame:
    """
    Expand specimen-level tooth-class rows into per-tooth observations.

    Each original row with `sockets` total teeth and `num_amtl` missing teeth
    is expanded into `sockets` rows: `num_amtl` with amtl=1 and the rest 0.
    """
    rows = []
    cols_to_keep = [
        "specimen",
        "genus",
        "tooth_class",
        "age",
        "prob_male",
        "is_human",
    ]
    for _, row in df.iterrows():
        n_missing = int(row["num_amtl"])
        n_sockets = int(row["sockets"])
        n_present = n_sockets - n_missing
        if n_sockets <= 0:
            continue

        base = {c: row[c] for c in cols_to_keep}

        # Missing teeth
        for _ in range(max(n_missing, 0)):
            r = base.copy()
            r["amtl"] = 1
            rows.append(r)

        # Present teeth
        for _ in range(max(n_present, 0)):
            r = base.copy()
            r["amtl"] = 0
            rows.append(r)

    return pd.DataFrame(rows)


def main() -> None:
    df = pd.read_csv("amtl.csv")

    # Keep only rows with genera relevant to the research question
    target_genera = ["Homo sapiens", "Pan", "Pongo", "Papio"]
    df = df[df["genus"].isin(target_genera)].copy()

    # Drop rows where sockets is zero or missing to avoid invalid trials
    df = df[df["sockets"] > 0].copy()

    # Indicator for modern humans vs non-human primates
    df["is_human"] = (df["genus"] == "Homo sapiens").astype(int)

    # Expand to per-tooth data for standard logistic regression
    long_df = expand_to_teeth(df)

    # Logistic regression at the tooth level: probability that a given tooth
    # is missing (amtl=1) as a function of human status, age, sex (prob_male),
    # and tooth class. Use cluster-robust SEs by specimen.
    model = smf.logit(
        formula="amtl ~ is_human + age + prob_male + C(tooth_class)",
        data=long_df,
    )
    result = model.fit(disp=False, cov_type="cluster", cov_kwds={"groups": long_df["specimen"]})

    is_human_coef = result.params["is_human"]
    is_human_pval = result.pvalues["is_human"]
    conf_int = result.conf_int().loc["is_human"]

    # Average predicted probability of AMTL for humans vs non-humans
    long_df = long_df.copy()
    long_df["pred"] = result.predict(long_df)
    pred_by_group = long_df.groupby("is_human")["pred"].mean()
    pred_nonhuman = float(pred_by_group.get(0, float("nan")))
    pred_human = float(pred_by_group.get(1, float("nan")))

    print("Logistic regression of AMTL at the tooth level")
    print(result.summary())
    print("\nEffect of being human (is_human):")
    print(f"  Coefficient (log-odds): {is_human_coef:.3f}")
    print(f"  95% CI: [{conf_int[0]:.3f}, {conf_int[1]:.3f}]")
    print(f"  p-value: {is_human_pval:.4g}")
    print("\nMean predicted probability of AMTL per tooth:")
    print(f"  Humans:     {pred_human:.4f}")
    print(f"  Non-humans: {pred_nonhuman:.4f}")


if __name__ == "__main__":
    main()
