import pandas as pd
import statsmodels.api as sm


def main() -> None:
    df = pd.read_csv("crofoot.csv")

    # Outcome: 1 if focal group won, 0 otherwise
    y = df["m_focal"]

    # Relative group size: focal size minus other size
    # According to metadata, f_other and win contain total group sizes
    rel_size = df["f_other"] - df["win"]

    # Relative location advantage:
    # Smaller distance to the center of its own home range indicates stronger "ownership".
    # Positive values mean the focal group is closer to its center than the other group is to its own.
    rel_loc = df["n_focal"] - df["m_other"]

    X = pd.DataFrame(
        {
            "rel_size": rel_size,
            "rel_loc": rel_loc,
        }
    )
    X = sm.add_constant(X)

    logit_model = sm.Logit(y, X)
    result = logit_model.fit(disp=False)

    # Store key statistics needed for interpretation
    params = result.params
    pvalues = result.pvalues

    # Simple heuristic to map statistical evidence to a scalar in [-100, 100]
    # Start from 0 (neutral) and add contributions from each predictor.
    score = 0.0

    for name in ["rel_size", "rel_loc"]:
        p = float(pvalues[name])
        coef = float(params[name])

        # Direction: positive effect supports "yes" that the factor increases win probability.
        direction = 1.0 if coef > 0 else -1.0

        if p < 0.001:
            contribution = 40.0
        elif p < 0.01:
            contribution = 30.0
        elif p < 0.05:
            contribution = 20.0
        elif p < 0.1:
            contribution = 10.0
        else:
            contribution = 0.0

        score += direction * contribution

    # Clamp to [-100, 100] and round to nearest integer
    score = max(-100, min(100, score))
    scalar = int(round(score))

    with open("conclusion.txt", "w", encoding="utf-8") as f:
        f.write(str(scalar))


if __name__ == "__main__":
    main()

