import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


def load_metadata(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_data(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def compute_effect_strength(df: pd.DataFrame) -> tuple[float, float]:
    """
    Compute the Pearson correlation between student-teacher ratio and
    academic performance (average of reading and math scores).

    Returns
    -------
    r : float
        Pearson correlation between ratio and performance.
    p : float
        Two-sided p-value.
    """
    # Student-teacher ratio: students per teacher
    ratio = df["students"] / df["teachers"]

    # Academic performance: average of reading and math scores
    performance = (df["read"] + df["math"]) / 2.0

    # Drop any rows with missing or infinite values
    mask = np.isfinite(ratio) & np.isfinite(performance)
    ratio_clean = ratio[mask]
    performance_clean = performance[mask]

    if ratio_clean.empty:
        return np.nan, np.nan

    r, p = stats.pearsonr(ratio_clean, performance_clean)
    return float(r), float(p)


def map_to_likert(r: float, p: float) -> int:
    """
    Map the correlation result to a Likert-style scalar from -100 to 100
    answering:

    \"Is a lower student-teacher ratio associated with higher academic performance?\"

    Positive values indicate evidence in favor of the statement,
    negative values indicate evidence against it.
    """
    if not np.isfinite(r) or not np.isfinite(p):
        return 0

    # Effect is defined so that positive values support the research question.
    # Lower ratios (more teachers per student) imply higher performance
    # when the correlation between ratio and performance is negative.
    effect = -r  # positive effect => support for the research question

    # Determine maximum achievable magnitude based on statistical significance
    if p > 0.10:
        max_magnitude = 20  # essentially weak, possibly spurious
    elif p > 0.05:
        max_magnitude = 30  # marginal evidence
    elif p > 0.01:
        max_magnitude = 60  # reasonably strong evidence
    else:
        max_magnitude = 90  # very strong evidence

    # Scale by the absolute size of the effect (correlation in [-1, 1])
    magnitude = min(abs(effect), 1.0)
    scalar = int(round(max_magnitude * magnitude))

    # Apply sign so that positive means "Yes", negative means "No"
    scalar = scalar if effect >= 0 else -scalar

    # Safety clip to the allowed range
    return int(max(-100, min(100, scalar)))


def main() -> None:
    base_dir = Path(__file__).parent

    # Load metadata (not strictly needed for the computation, but adheres to instructions)
    info_path = base_dir / "info.json"
    if info_path.exists():
        _ = load_metadata(info_path)

    data_path = base_dir / "caschools.csv"
    df = load_data(data_path)

    r, p = compute_effect_strength(df)
    scalar = map_to_likert(r, p)

    # Write the scalar conclusion to file, as required
    conclusion_path = base_dir / "conclusion.txt"
    with conclusion_path.open("w", encoding="utf-8") as f:
        f.write(f"{scalar}")


if __name__ == "__main__":
    main()

