import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency


def cramers_v(chi2: float, n: int, r: int, c: int) -> float:
    """Compute Cramer's V effect size."""
    if n <= 0:
        return 0.0
    k = min(r, c)
    if k <= 1:
        return 0.0
    return math.sqrt(chi2 / (n * (k - 1)))


def effect_from_table(table: pd.DataFrame) -> tuple[float, float]:
    """Return (p_value, Cramer's V) from a contingency table."""
    chi2, p, dof, expected = chi2_contingency(table)
    v = cramers_v(chi2, int(table.to_numpy().sum()), *table.shape)
    return float(p), float(v)


def map_effect_to_scalar(vs: list[float]) -> int:
    """
    Map average Cramer's V to Likert [-100, 100].

    - 0   -> 0 (no variation)
    - 0.3 -> ~60 (moderate)
    - 0.5 -> 100 (very strong)
    Values are clipped to [0, 100] and kept non-negative since we're
    quantifying strength of evidence that variation exists.
    """
    if not vs:
        return 0
    avg_v = float(np.mean(vs))
    # Reference V=0.5 as "maximal" for our scale.
    scaled = (avg_v / 0.5) * 100.0
    scalar = max(0.0, min(100.0, scaled))
    return int(round(scalar))


def main() -> None:
    data_path = Path("boxes.csv")
    df = pd.read_csv(data_path)

    # Outcome: 1=unchosen, 2=majority, 3=minority
    outcome = df["feature1"]

    # Variation across cultural sites.
    tbl_site = pd.crosstab(outcome, df["feature5"])
    p_site, v_site = effect_from_table(tbl_site)

    # Define age groups (developmental stages) via quantiles to avoid
    # arbitrary cut-points while still reflecting early/mid/late ranges.
    df["age_group"] = pd.qcut(df["feature3"], q=3, labels=["younger", "middle", "older"])
    tbl_age = pd.crosstab(outcome, df["age_group"])
    p_age, v_age = effect_from_table(tbl_age)

    scalar = map_effect_to_scalar([v_site, v_age])

    # For debugging/interpretation in logs, but not in conclusion.txt.
    print("Site effect:   p =", p_site, "V =", v_site)
    print("Age effect:    p =", p_age, "V =", v_age)
    print("Scalar answer:", scalar)

    # Write final scalar to conclusion.txt with no extra text.
    Path("conclusion.txt").write_text(str(scalar), encoding="utf-8")


if __name__ == "__main__":
    main()

