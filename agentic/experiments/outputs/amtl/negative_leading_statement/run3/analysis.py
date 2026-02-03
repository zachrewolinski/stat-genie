import pandas as pd
import numpy as np
import statsmodels.api as sm
import patsy

def fit_model(df):
    # Filter any impossible rows
    df = df.copy()
    df = df[df["sockets"] > 0]
    df = df[df["num_amtl"].between(0, df["sockets"])]

    df["human"] = (df["genus"] == "Homo sapiens").astype(int)

    # Build design matrix
    formula = "human + age + prob_male + C(tooth_class)"
    X = patsy.dmatrix(formula, df, return_type="dataframe")
    design_info = X.design_info
    y = np.column_stack([df["num_amtl"].to_numpy(), (df["sockets"] - df["num_amtl"]).to_numpy()])

    model = sm.GLM(y, X, family=sm.families.Binomial())
    result = model.fit()
    return result, df, design_info


def predicted_rate(result, df, design_info, tooth_class_weights=None):
    # Predict at mean age and prob_male, averaged across tooth classes
    mean_age = df["age"].mean()
    mean_prob_male = df["prob_male"].mean()

    classes = sorted(df["tooth_class"].unique())
    if tooth_class_weights is None:
        counts = df["tooth_class"].value_counts().reindex(classes)
        weights = counts / counts.sum()
    else:
        weights = pd.Series(tooth_class_weights).reindex(classes).fillna(0)
        weights = weights / weights.sum()

    def build_row(human, tooth_class):
        row = {
            "human": human,
            "age": mean_age,
            "prob_male": mean_prob_male,
            "tooth_class": tooth_class,
        }
        return row

    pred = {}
    for human in [0, 1]:
        probs = []
        for tc in classes:
            row_df = pd.DataFrame([build_row(human, tc)])
            X_new = patsy.dmatrix(design_info, row_df, return_type="dataframe")
            p = result.predict(X_new)[0]
            probs.append(p)
        pred[human] = float((weights.values * np.array(probs)).sum())
    return pred


def main():
    df = pd.read_csv("amtl.csv")

    result, df, design_info = fit_model(df)

    pred = predicted_rate(result, df, design_info)
    diff = pred[1] - pred[0]
    ratio = pred[1] / pred[0] if pred[0] > 0 else np.nan

    print(result.summary())
    print("\nPredicted AMTL rate (mean covariates, weighted tooth_class):")
    print(f"Non-human: {pred[0]:.4f}")
    print(f"Human: {pred[1]:.4f}")
    print(f"Difference (Human - Non-human): {diff:.4f}")
    print(f"Ratio (Human / Non-human): {ratio:.4f}")
    print("\nHuman coefficient:")
    print(f"coef={result.params['human']:.4f}, p={result.pvalues['human']:.4g}")

if __name__ == "__main__":
    main()
