import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def load_metadata(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def main() -> None:
    base_dir = Path(__file__).parent
    info_path = base_dir / "info.json"
    data_path = base_dir / "amtl.csv"

    info = load_metadata(info_path)
    print("Research question:")
    for q in info.get("research_questions", []):
        print(f" - {q}")
    print()

    df = pd.read_csv(data_path)

    # Compute AMTL proportion
    df = df.copy()
    df["amtl_prop"] = df["num_amtl"] / df["sockets"]

    # Define human vs non-human indicator based on genus
    df["is_human"] = df["genus"].astype(str).str.contains("Homo", case=False, na=False).astype(int)

    # Keep only rows with non-missing key variables
    key_cols = ["num_amtl", "sockets", "age", "prob_male", "tooth_class", "is_human"]
    df_model = df.dropna(subset=key_cols).copy()

    # Descriptive statistics
    genus_summary = (
        df_model.groupby("genus")
        .agg(
            n_specimens=("specimen", "nunique"),
            n_rows=("specimen", "size"),
            mean_amtl_prop=("amtl_prop", "mean"),
        )
        .reset_index()
    )

    print("Descriptive AMTL proportion by genus:")
    print(genus_summary.to_string(index=False))
    print()

    # Binomial regression: num_amtl successes out of sockets trials
    # Model: logit(p) = beta0 + beta1 * is_human + beta2 * age + beta3 * prob_male + tooth_class effects
    # Use Homo vs all non-human genera; adjust for age, sex, and tooth class.
    exog_vars = ["is_human", "age", "prob_male"]
    tooth_dummies = pd.get_dummies(df_model["tooth_class"], prefix="tooth", drop_first=True)
    exog = pd.concat([df_model[exog_vars], tooth_dummies], axis=1)
    exog = sm.add_constant(exog, has_constant="add")

    endog = np.asarray(
        np.column_stack(
            [
                df_model["num_amtl"].to_numpy(),
                (df_model["sockets"] - df_model["num_amtl"]).to_numpy(),
            ]
        )
    )

    model = sm.GLM(endog, exog, family=sm.families.Binomial())
    result = model.fit()

    print("Binomial regression results (AMTL probability):")
    print(result.summary())

    # Extract the is_human effect for easier reference
    if "is_human" in result.params:
        beta_human = result.params["is_human"]
        se_human = result.bse["is_human"]
        p_human = result.pvalues["is_human"]
        print("\nEffect of being human (is_human):")
        print(f"  Coefficient (log-odds): {beta_human:.4f}")
        print(f"  Std. error           : {se_human:.4f}")
        print(f"  p-value              : {p_human:.4g}")

        # Approximate odds ratio
        or_human = float(np.exp(beta_human))
        print(f"  Odds ratio           : {or_human:.3f}")


if __name__ == "__main__":
    main()

