import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def main() -> None:
    info_path = Path("info.json")
    data_path = Path("crofoot.csv")

    with info_path.open("r", encoding="utf-8") as f:
        info = json.load(f)

    research_question = info.get("research_questions", [""])[0]

    df = pd.read_csv(data_path)

    df["size_diff"] = df["feature7"] - df["feature8"]
    df["loc_diff"] = df["feature5"] - df["feature6"]

    y = df["feature4"]
    X = df[["size_diff", "loc_diff"]]
    X = sm.add_constant(X, has_constant="add")

    model = sm.Logit(y, X)
    result = model.fit(disp=False)

    print("Research question:")
    print(research_question)
    print()
    print("Logistic regression summary:")
    print(result.summary())

    pvalues = result.pvalues

    def p_to_score(p: float) -> int:
        if p <= 0.001:
            return 40
        if p <= 0.01:
            return 30
        if p <= 0.05:
            return 20
        if p <= 0.1:
            return 10
        if p <= 0.2:
            return 5
        return 0

    size_p = float(pvalues.get("size_diff", np.nan))
    loc_p = float(pvalues.get("loc_diff", np.nan))

    size_score = p_to_score(size_p) if np.isfinite(size_p) else 0
    loc_score = p_to_score(loc_p) if np.isfinite(loc_p) else 0

    lr_p = float(getattr(result, "llr_pvalue", np.nan))
    model_score = p_to_score(lr_p) if np.isfinite(lr_p) else 0

    total_score = size_score + loc_score + model_score

    max_raw = 120
    scaled = int(round((total_score / max_raw) * 100)) if max_raw > 0 else 0

    likert_scalar = max(-100, min(100, scaled))

    print()
    print("Derived evidence scores:")
    print(f"  size_diff p={size_p:.4g}, score={size_score}")
    print(f"  loc_diff  p={loc_p:.4g}, score={loc_score}")
    print(f"  LR test  p={lr_p:.4g}, score={model_score}")
    print(f"  Total evidence score={total_score} -> Likert scalar={likert_scalar}")

    conclusion_path = Path("conclusion.txt")
    with conclusion_path.open("w", encoding="utf-8") as f:
        f.write(str(likert_scalar))


if __name__ == "__main__":
    main()

