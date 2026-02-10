import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def main() -> None:
    base_path = Path(".")

    info_path = base_path / "info.json"
    data_path = base_path / "boxes.csv"
    conclusion_path = base_path / "conclusion.txt"

    # Load metadata (not strictly needed for the computation, but for completeness)
    with info_path.open("r", encoding="utf-8") as f:
        info = json.load(f)

    df = pd.read_csv(data_path)

    # Outcome encoding: 1 = unchosen, 2 = majority, 3 = minority
    df_demo = df[df["y"].isin([2, 3])].copy()
    df_demo["majority_choice"] = (df_demo["y"] == 2).astype(int)

    # Basic descriptive stats: overall majority preference
    overall_majority_rate = df_demo["majority_choice"].mean()

    # Create coarse developmental stages via age terciles
    # (robust to specific age scale while capturing variation across development)
    try:
        df_demo["age_group"] = pd.qcut(df_demo["age"], q=3, labels=["young", "mid", "old"])
    except ValueError:
        # Fallback in rare degenerate cases: use median split
        df_demo["age_group"] = pd.qcut(df_demo["age"], q=2, labels=["young", "old"])

    # Logistic regression: majority_choice ~ age + culture (categorical)
    # This directly targets whether majority reliance varies with age and culture.
    model = smf.logit("majority_choice ~ age + C(culture)", data=df_demo).fit(disp=False)

    pvalues = model.pvalues

    # Extract p-value for age (developmental trend)
    age_p = float(pvalues.get("age", np.nan))

    # Extract p-values for culture indicators (cross-cultural variation)
    culture_ps = [
        float(p)
        for name, p in pvalues.items()
        if name.startswith("C(culture)[T.")
    ]

    # Heuristic evidence score based on significance of age and culture effects.
    def evidence_component(p: float | None) -> float:
        if p is None or np.isnan(p):
            return 0.0
        if p < 1e-6:
            return 1.0
        if p < 1e-3:
            return 0.8
        if p < 1e-2:
            return 0.5
        if p < 5e-2:
            return 0.2
        return 0.0

    age_evidence = evidence_component(age_p)
    culture_evidence = 0.0
    if culture_ps:
        culture_evidence = max(evidence_component(p) for p in culture_ps)

    # Also incorporate effect size via spread of majority rates across age_group and culture.
    grp = df_demo.groupby(["age_group", "culture"])["majority_choice"].mean()
    variability = grp.std(ddof=0)

    # Map variability into [0, 1] with a soft cap; typical behavioral datasets
    # rarely exceed SD ~ 0.25 in proportions across groups.
    variability_component = min(variability / 0.25, 1.0) if not np.isnan(variability) else 0.0

    # Combine components: significance of age and culture plus variability in behavior.
    combined_evidence = 0.4 * age_evidence + 0.4 * culture_evidence + 0.2 * variability_component

    # Anchor on the fact that overall majority following exists at all; if children
    # generally follow the majority (overall rate > 0.5), that supports the idea
    # that majority cues are meaningful.
    if overall_majority_rate > 0.5:
        combined_evidence = min(combined_evidence + 0.1, 1.0)

    # Convert combined evidence in [0,1] to Likert scale [-100, 100],
    # where positive means "yes, there is variation across cultures/development".
    scalar_float = (combined_evidence * 2.0 - 1.0) * 100.0

    # Clip to bounds and round to nearest integer.
    scalar_int = int(np.clip(np.round(scalar_float), -100, 100))

    # Write scalar conclusion to file, as required (single integer, no extra text).
    conclusion_path.write_text(str(scalar_int), encoding="utf-8")


if __name__ == "__main__":
    main()

