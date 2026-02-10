import pandas as pd
import numpy as np


def main() -> None:
    df = pd.read_csv("crofoot.csv")

    # Outcome and predictors based on metadata
    win = df["feature4"]

    # Relative group size: focal minus other
    rel_size = df["feature7"] - df["feature8"]

    # Relative home-range proximity: other distance minus focal distance
    # Positive values mean focal is closer to its home-range center
    rel_location = df["feature6"] - df["feature5"]

    # Standardize predictors for comparability
    rel_size_z = (rel_size - rel_size.mean()) / rel_size.std(ddof=0)
    rel_location_z = (rel_location - rel_location.mean()) / rel_location.std(ddof=0)

    # Simple logistic-like score: linear probability model with two predictors
    # Fit via least squares: p_hat = b0 + b1 * size_z + b2 * loc_z
    X = np.column_stack(
        [
            np.ones(len(df)),
            rel_size_z.to_numpy(),
            rel_location_z.to_numpy(),
        ]
    )
    y = win.to_numpy()

    beta, *_ = np.linalg.lstsq(X, y, rcond=None)

    b0, b_size, b_loc = beta

    # Measure overall predictive strength using pseudo-R^2 (variance explained)
    y_hat = X @ beta
    ss_tot = ((y - y.mean()) ** 2).sum()
    ss_res = ((y - y_hat) ** 2).sum()
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    # Combine directionality and strength into a single effect score.
    # Direction from coefficients; strength from R^2.
    # We weight size and location effects equally.
    direction_score = np.tanh(b_size) + np.tanh(b_loc)
    max_direction = 2.0  # since each tanh term in [-1, 1]
    normalized_direction = direction_score / max_direction  # in [-1, 1]

    # Scale strength into [0, 1] but emphasize moderate R^2
    strength = min(max(r2, 0.0), 1.0)

    # Final scalar in [-100, 100]
    scalar = int(round(100 * normalized_direction * strength))

    # Write scalar only to conclusion.txt
    with open("conclusion.txt", "w", encoding="utf-8") as f:
        f.write(str(scalar))


if __name__ == "__main__":
    main()

