import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main() -> None:
    df = pd.read_csv("amtl.csv")

    # Basic cleaning: drop rows with missing key fields
    key_cols = ["num_amtl", "sockets", "age", "prob_male", "tooth_class", "genus"]
    df = df.dropna(subset=key_cols).copy()

    # Create human vs non-human indicator
    df["is_human"] = df["genus"].astype(str).str.contains("Homo", case=False).astype(int)

    # Remove any rows with non-positive socket counts to avoid invalid binomial denominators
    df = df[df["sockets"] > 0].copy()

    # Response as proportion with binomial weights
    df["amtl_prop"] = df["num_amtl"] / df["sockets"]

    # Fit binomial regression: AMTL ~ human vs non-human + age + sex + tooth class
    formula = "amtl_prop ~ is_human + age + prob_male + C(tooth_class)"

    model = smf.glm(
        formula=formula,
        data=df,
        family=sm.families.Binomial(),
        freq_weights=df["sockets"],
    )
    result = model.fit()

    coef = float(result.params["is_human"])
    pval = float(result.pvalues["is_human"])
    oratio = float(np.exp(coef))

    # Counterfactual predictions toggling human vs non-human while holding other covariates fixed
    df_human = df.copy()
    df_human["is_human"] = 1
    df_nonhuman = df.copy()
    df_nonhuman["is_human"] = 0

    pred_human = result.predict(df_human)
    pred_nonhuman = result.predict(df_nonhuman)

    mean_pred_human = float(np.mean(pred_human))
    mean_pred_nonhuman = float(np.mean(pred_nonhuman))

    # Overall observed proportions for reference
    total_amtl = df["num_amtl"].sum()
    total_sockets = df["sockets"].sum()
    overall_prop = float(total_amtl / total_sockets)

    total_amtl_human = df.loc[df["is_human"] == 1, "num_amtl"].sum()
    total_sockets_human = df.loc[df["is_human"] == 1, "sockets"].sum()
    prop_human = float(total_amtl_human / total_sockets_human)

    total_amtl_nonhuman = df.loc[df["is_human"] == 0, "num_amtl"].sum()
    total_sockets_nonhuman = df.loc[df["is_human"] == 0, "sockets"].sum()
    prop_nonhuman = float(total_amtl_nonhuman / total_sockets_nonhuman)

    print(f"is_human_coef={coef}")
    print(f"is_human_p={pval}")
    print(f"is_human_or={oratio}")
    print(f"mean_pred_human={mean_pred_human}")
    print(f"mean_pred_nonhuman={mean_pred_nonhuman}")
    print(f"overall_prop={overall_prop}")
    print(f"obs_prop_human={prop_human}")
    print(f"obs_prop_nonhuman={prop_nonhuman}")
    print(f"n_rows={len(df)}")
    print(f"n_human={int(df['is_human'].sum())}")
    print(f"n_nonhuman={int((1 - df['is_human']).sum())}")


if __name__ == "__main__":
    main()

