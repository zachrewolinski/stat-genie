import json
from pathlib import Path

import numpy as np
import pandas as pd
import patsy
import statsmodels.api as sm
from scipy import stats


def main() -> None:
    base_dir = Path(__file__).parent

    info_path = base_dir / "info.json"
    data_path = base_dir / "amtl.csv"

    with info_path.open("r") as f:
        info = json.load(f)

    research_question = info.get("research_questions", [""])[0]

    df = pd.read_csv(data_path)

    # Basic cleaning / sanity checks
    df = df.copy()
    df = df[df["sockets"] > 0].reset_index(drop=True)

    # Drop any logically invalid rows where num_amtl exceeds available sockets
    invalid_mask = df["num_amtl"] > df["sockets"]
    if invalid_mask.any():
        print(
            f"Dropping {invalid_mask.sum()} rows where num_amtl > sockets "
            "because these are invalid for a binomial model."
        )
        df = df[~invalid_mask].reset_index(drop=True)

    # Create response proportion for quick summaries
    df["prop_amtl"] = df["num_amtl"] / df["sockets"]

    # Quick descriptive statistics by genus
    genus_summary = (
        df.groupby("genus")
        .agg(
            mean_prop=("prop_amtl", "mean"),
            median_prop=("prop_amtl", "median"),
            n_rows=("prop_amtl", "size"),
        )
        .sort_values("mean_prop", ascending=False)
    )

    print("Research question:")
    print(research_question)
    print("\nGenus-level descriptive statistics (higher = more AMTL):")
    print(genus_summary)

    # Binomial regression using aggregated counts with sockets as trials.
    # We model logit(p(AMTL)) ~ genus + age + prob_male + tooth_class.
    # Represent the response as [successes, failures] for each row.
    y = np.column_stack(
        [
            df["num_amtl"].astype(float),
            (df["sockets"] - df["num_amtl"]).astype(float),
        ]
    )

    formula_full = "C(genus) + age + prob_male + C(tooth_class)"
    X_full = patsy.dmatrix(formula_full, df, return_type="dataframe")
    model_full = sm.GLM(y, X_full, family=sm.families.Binomial()).fit()

    # Model without genus to test its joint contribution
    formula_nogenus = "age + prob_male + C(tooth_class)"
    X_nogenus = patsy.dmatrix(formula_nogenus, df, return_type="dataframe")
    model_nogenus = sm.GLM(y, X_nogenus, family=sm.families.Binomial()).fit()

    # Likelihood ratio test for genus as a block
    lr_stat = model_nogenus.deviance - model_full.deviance
    df_diff = model_nogenus.df_resid - model_full.df_resid
    p_lr = stats.chi2.sf(lr_stat, df_diff)

    print("\n=== Binomial regression results (AMTL proportion) ===")
    print(model_full.summary())
    print("\nLikelihood ratio test for adding genus:")
    print(f"  LR statistic = {lr_stat:.3f}, df = {df_diff}, p-value = {p_lr:.3e}")

    # Extract genus coefficients (Homo sapiens is baseline category by default)
    params = model_full.params
    conf_int = model_full.conf_int()

    genus_terms = [name for name in params.index if name.startswith("C(genus)[")]
    print("\nGenus effects (log-odds relative to Homo sapiens):")
    for term in genus_terms:
        coef = params[term]
        ci_low, ci_high = conf_int.loc[term]
        odds_ratio = float(np.exp(coef))
        or_low = float(np.exp(ci_low))
        or_high = float(np.exp(ci_high))
        print(
            f"  {term}: coef = {coef:.3f}, OR = {odds_ratio:.3f} "
            f"(95% CI {or_low:.3f}–{or_high:.3f})"
        )

    # Predict AMTL proportions at representative covariate values
    # Use median age, median prob_male, and the most common tooth_class.
    median_age = df["age"].median()
    median_prob_male = df["prob_male"].median()
    common_tooth_class = df["tooth_class"].mode().iat[0]

    genera = sorted(df["genus"].unique())
    pred_rows = []
    for g in genera:
        pred_rows.append(
            {
                "genus": g,
                "age": median_age,
                "prob_male": median_prob_male,
                "tooth_class": common_tooth_class,
            }
        )
    pred_df = pd.DataFrame(pred_rows)
    # Use the original design_info to ensure column alignment with the fitted model
    X_pred = patsy.build_design_matrices(
        [X_full.design_info], pred_df
    )[0]
    preds = model_full.get_prediction(X_pred)
    pred_summary = preds.summary_frame(alpha=0.05)
    pred_df["predicted_prop"] = pred_summary["mean"]
    pred_df["ci_lower"] = pred_summary["mean_ci_lower"]
    pred_df["ci_upper"] = pred_summary["mean_ci_upper"]
    pred_df = pred_df.sort_values("predicted_prop", ascending=False)

    print("\nPredicted AMTL proportion at median age/sex and common tooth class:")
    print(pred_df)


if __name__ == "__main__":
    main()
