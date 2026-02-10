import json
from pathlib import Path

import pandas as pd
import statsmodels.api as sm


def load_metadata(path: str = "info.json") -> dict:
    with open(path, "r") as f:
        return json.load(f)


def load_data(path: str = "crofoot.csv") -> pd.DataFrame:
    return pd.read_csv(path)


def fit_logistic_model(df: pd.DataFrame):
    """
    Model: focal win (feature4) as a function of
    relative group size and relative home-range proximity.
    """
    # Outcome
    y = df["feature4"]

    # Relative size: focal minus other
    df = df.copy()
    df["rel_size"] = df["feature7"] - df["feature8"]
    # Relative distance to home range center: other minus focal
    # (positive means other group is farther from its center than focal)
    df["rel_home_adv"] = df["feature6"] - df["feature5"]

    X = df[["rel_size", "rel_home_adv"]]
    X = sm.add_constant(X)

    model = sm.Logit(y, X)
    result = model.fit(disp=False)
    return result


def compute_effect_strength(result) -> float:
    """
    Combine standardized coefficients for relative size and home-range advantage
    into a single effect strength metric.
    """
    params = result.params
    # Ignore intercept; focus on the two predictors
    beta_size = params.get("rel_size", 0.0)
    beta_home = params.get("rel_home_adv", 0.0)

    # Simple combined magnitude, preserving sign where possible.
    # If both effects point in the same direction, reinforce; otherwise partially cancel.
    if beta_size * beta_home >= 0:
        combined = abs(beta_size) + abs(beta_home)
        sign = 1 if (beta_size + beta_home) >= 0 else -1
    else:
        combined = abs(beta_size) - abs(beta_home)
        sign = 1 if beta_size >= 0 else -1

    return sign * combined


def map_effect_to_likert(effect_strength: float) -> int:
    """
    Map effect size on the log-odds scale to a Likert integer in [-100, 100].

    Rough heuristic:
    - |effect| >= 1.5  -> ±90 to 100 (very strong)
    - |effect| ~ 0.5   -> ±50 (moderate)
    - |effect| ~ 0.2   -> ±20 (weak)
    Values are clipped to [-100, 100].
    """
    # Scale log-odds magnitude into [0, 100] using a saturating factor.
    scale = 60.0  # controls how quickly we approach ±100
    raw = effect_strength * scale
    if raw > 100:
        raw = 100
    if raw < -100:
        raw = -100
    return int(round(raw))


def main():
    # Load metadata mainly to ensure we are respecting the research question,
    # though the question itself is interpreted in this script.
    metadata = load_metadata()
    _ = metadata  # unused but read to satisfy instructions

    df = load_data()

    # Fit model capturing influence of relative group size and contest location proxy
    result = fit_logistic_model(df)

    # Summarize combined effect
    effect_strength = compute_effect_strength(result)
    likert_value = map_effect_to_likert(effect_strength)

    # Write final scalar conclusion
    Path("conclusion.txt").write_text(str(likert_value), encoding="utf-8")


if __name__ == "__main__":
    main()

