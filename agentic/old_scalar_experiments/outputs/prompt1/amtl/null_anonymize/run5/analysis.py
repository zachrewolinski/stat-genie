import json
from pathlib import Path

import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def load_data(csv_path: str = "amtl.csv") -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df.rename(
        columns={
            "feature1": "tooth_class",
            "feature3": "n_missing",
            "feature4": "n_sockets",
            "feature5": "age",
            "feature7": "sex_est",
            "feature8": "genus",
        }
    )
    df = df[df["n_sockets"] > 0].copy()
    df["prop_missing"] = df["n_missing"] / df["n_sockets"]
    df["genus"] = df["genus"].astype("category")
    df["tooth_class"] = df["tooth_class"].astype("category")
    return df


def fit_binomial_model(df: pd.DataFrame):
    model = smf.glm(
        formula="prop_missing ~ C(genus) + age + sex_est + C(tooth_class)",
        data=df,
        family=sm.families.Binomial(),
        freq_weights=df["n_sockets"],
    ).fit()
    return model


def summarize_genus_effects(df: pd.DataFrame, model) -> dict:
    genus_summary = (
        df.groupby("genus")
        .agg(total_missing=("n_missing", "sum"), total_sockets=("n_sockets", "sum"))
        .assign(amtl_rate=lambda g: g["total_missing"] / g["total_sockets"])
    )

    params = model.params
    conf = model.conf_int()

    effects = {}
    for genus in genus_summary.index:
        if genus == "Homo sapiens":
            continue
        param_name = f"C(genus)[T.{genus}]"
        if param_name not in params.index:
            continue
        coef = float(params[param_name])
        ci_low, ci_high = map(float, conf.loc[param_name])
        effects[genus] = {
            "coef_vs_human": coef,
            "ci_low": ci_low,
            "ci_high": ci_high,
        }

    return {
        "genus_summary": genus_summary.reset_index().to_dict(orient="records"),
        "effects_vs_human": effects,
    }


def main():
    df = load_data()
    model = fit_binomial_model(df)

    summary = summarize_genus_effects(df, model)

    print("Per-genus AMTL rates (weighted by sockets):")
    for row in summary["genus_summary"]:
        print(
            f"{row['genus']}: missing={row['total_missing']}, "
            f"sockets={row['total_sockets']}, "
            f"rate={row['amtl_rate']:.3f}"
        )

    print("\nGenus coefficients relative to Homo sapiens (baseline expected):")
    for genus, eff in summary["effects_vs_human"].items():
        print(
            f"{genus}: coef={eff['coef_vs_human']:.3f}, "
            f"95% CI [{eff['ci_low']:.3f}, {eff['ci_high']:.3f}]"
        )

    # Optionally store a machine-readable summary alongside printed output.
    Path("model_summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

