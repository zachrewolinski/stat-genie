import pandas as pd
import numpy as np
import statsmodels.api as sm


def main() -> None:
    df = pd.read_csv("crofoot.csv")

    # Outcome: focal group win (1) vs loss (0)
    y = df["feature4"].astype(float)

    # Relative group size (focal - other) and relative distance (focal - other)
    df["rel_size"] = df["feature7"] - df["feature8"]
    df["rel_dist"] = df["feature5"] - df["feature6"]

    X = df[["rel_size", "rel_dist"]]
    X = sm.add_constant(X)

    logit_model = sm.Logit(y, X)
    try:
        res = logit_model.fit(disp=False)
    except Exception:
        # In case of complete separation or other small-sample issues,
        # fall back to simple correlations as a heuristic.
        rel_size_effect = np.corrcoef(df["rel_size"], y)[0, 1]
        rel_dist_effect = np.corrcoef(df["rel_dist"], y)[0, 1]
    else:
        rel_size_effect = res.params["rel_size"]
        rel_dist_effect = res.params["rel_dist"]

    # Interpret effects:
    # Positive rel_size_effect -> larger relative group size increases win probability.
    # Negative rel_dist_effect -> being closer to home (smaller distance) increases win probability.

    # Normalize effects into [-1, 1] range using a simple tanh to bound extremes.
    size_signal = np.tanh(rel_size_effect)
    # For distance, flip sign so that positive means advantage when closer to home.
    dist_signal = np.tanh(-rel_dist_effect)

    # Combine signals: give equal weight to size and location.
    combined_signal = 0.5 * size_signal + 0.5 * dist_signal

    # Map combined signal in [-1, 1] to Likert scale [-100, 100]
    scalar = int(np.round(combined_signal * 100))

    # Ensure scalar is within bounds
    scalar = max(-100, min(100, scalar))

    with open("conclusion.txt", "w") as f:
        f.write(str(scalar))


if __name__ == "__main__":
    main()

