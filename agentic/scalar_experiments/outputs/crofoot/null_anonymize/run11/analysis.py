import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats


def main() -> None:
    base_dir = Path(__file__).parent

    # Load metadata (not strictly needed for computation, but keeps context explicit)
    info_path = base_dir / "info.json"
    with info_path.open("r") as f:
        info = json.load(f)

    # Load dataset
    data_path = base_dir / "crofoot.csv"
    df = pd.read_csv(data_path)

    # Response: 1 if focal won, 0 otherwise
    y = df["feature4"].astype(float)

    # Predictors capturing relative group size and contest location
    # Relative group size: focal size minus other size
    size_diff = df["feature7"] - df["feature8"]

    # Contest location advantage: how much closer the focal group is to its home range center
    # Positive values mean focal is closer (other group is farther from its center).
    loc_adv = df["feature6"] - df["feature5"]

    X = pd.DataFrame(
        {
            "size_diff": size_diff.astype(float),
            "loc_adv": loc_adv.astype(float),
        }
    )
    X = sm.add_constant(X, has_constant="add")

    # Fit full logistic regression model
    try:
        full_model = sm.Logit(y, X).fit(disp=False)
    except Exception:
        # If the full model fails to converge for any reason, fall back to a very conservative neutral answer.
        scalar = 0
        (base_dir / "conclusion.txt").write_text(str(int(scalar)), encoding="utf-8")
        return

    # Fit null model with intercept only
    X_null = np.ones((len(y), 1), dtype=float)
    null_model = sm.Logit(y, X_null).fit(disp=False)

    # Likelihood ratio test for joint effect of size_diff and loc_adv
    lr_stat = 2.0 * (full_model.llf - null_model.llf)
    df_diff = full_model.df_model - null_model.df_model
    if df_diff <= 0:
        # Should not happen here, but guard anyway.
        scalar = 0
    else:
        p_value = stats.chi2.sf(lr_stat, df_diff)

        # Map p-value to Likert scale [-100, 100]:
        # p = 0   ->  strong "Yes"  (100)
        # p = 0.5 ->  neutral       (0)
        # p = 1   ->  strong "No"   (-100)
        scalar_float = (0.5 - p_value) * 200.0
        scalar = int(round(max(-100, min(100, scalar_float))))

    # Write final scalar to conclusion.txt with no extra text or lines.
    conclusion_path = base_dir / "conclusion.txt"
    conclusion_path.write_text(str(int(scalar)), encoding="utf-8")


if __name__ == "__main__":
    main()

