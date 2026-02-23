import pandas as pd
import numpy as np
import statsmodels.api as sm


def main() -> None:
    df = pd.read_csv("amtl.csv")

    # Keep rows with all required fields and positive socket counts
    required = ["num_amtl", "sockets", "age", "prob_male", "tooth_class", "genus"]
    df = df.dropna(subset=required)
    df = df[df["sockets"] > 0].copy()

    # Drop rows where counts exceed available sockets (biologically impossible; likely data errors)
    invalid_mask = df["num_amtl"] > df["sockets"]
    if invalid_mask.any():
        print(
            f"Dropping {invalid_mask.sum()} rows with num_amtl > sockets "
            "because they are inconsistent with the binomial model."
        )
        df = df.loc[~invalid_mask].copy()

    # Indicator for modern humans vs non-human primates
    df["is_human"] = (df["genus"] == "Homo sapiens").astype(int)

    # Descriptive AMTL prevalence by genus
    grouped = (
        df.groupby("genus")
        .apply(
            lambda g: pd.Series(
                {
                    "total_missing": g["num_amtl"].sum(),
                    "total_sockets": g["sockets"].sum(),
                    "prop_missing": g["num_amtl"].sum() / g["sockets"].sum(),
                    "n_rows": len(g),
                }
            )
        )
        .sort_values("prop_missing", ascending=False)
    )

    print("=== Descriptive AMTL prevalence by genus ===")
    print(grouped.to_string(float_format=lambda x: f"{x:0.4f}"))
    print()

    # Design matrix with tooth class as categorical (dummy-coded)
    X = pd.get_dummies(
        df[["is_human", "age", "prob_male", "tooth_class"]],
        columns=["tooth_class"],
        drop_first=True,
    )
    X = sm.add_constant(X)

    # Binomial response as [successes, failures]
    y = np.column_stack([df["num_amtl"], df["sockets"] - df["num_amtl"]])

    model = sm.GLM(y, X, family=sm.families.Binomial())
    result = model.fit()

    print("=== Binomial regression results (GLM) ===")
    print(result.summary())
    print()

    # Effect of being human: odds ratio, p-value, confidence interval
    human_coef = result.params["is_human"]
    human_p = result.pvalues["is_human"]
    human_ci_low, human_ci_high = result.conf_int().loc["is_human"]
    human_or = float(np.exp(human_coef))
    ci_or_low = float(np.exp(human_ci_low))
    ci_or_high = float(np.exp(human_ci_high))

    print("=== Human vs non-human effect ===")
    print(f"Log-odds coefficient (is_human): {human_coef:0.4f}")
    print(f"Odds ratio (human vs non-human): {human_or:0.3f}")
    print(
        "95% CI for odds ratio: "
        f"[{ci_or_low:0.3f}, {ci_or_high:0.3f}]"
    )
    print(f"p-value for is_human: {human_p:0.4g}")
    print()

    # Predicted AMTL probabilities for typical individuals
    mean_age = df["age"].mean()
    mean_prob_male = df["prob_male"].mean()
    tooth_classes = df["tooth_class"].unique()

    # Weight predictions by the distribution of sockets across tooth classes
    socket_weights = (
        df.groupby("tooth_class")["sockets"].sum()
        / df["sockets"].sum()
    )

    def predicted_overall_prob(is_human_value: int) -> float:
        probs = []
        for cls in tooth_classes:
            base = pd.DataFrame(
                {
                    "is_human": [is_human_value],
                    "age": [mean_age],
                    "prob_male": [mean_prob_male],
                    "tooth_class": [cls],
                }
            )
            X_pred = pd.get_dummies(
                base, columns=["tooth_class"], drop_first=True
            )
            # Align with training design matrix columns (excluding constant)
            X_pred = X_pred.reindex(columns=X.columns.drop("const"), fill_value=0)
            X_pred = sm.add_constant(X_pred, has_constant="add")
            p = float(result.predict(X_pred)[0])
            w = float(socket_weights.loc[cls])
            probs.append(p * w)
        return float(np.sum(probs))

    pred_nonhuman = predicted_overall_prob(0)
    pred_human = predicted_overall_prob(1)

    print("=== Predicted overall AMTL probabilities (adjusted) ===")
    print(f"Non-human primates: {pred_nonhuman:0.3%}")
    print(f"Modern humans:      {pred_human:0.3%}")
    print(f"Absolute difference: {pred_human - pred_nonhuman:0.3%}")


if __name__ == "__main__":
    main()
