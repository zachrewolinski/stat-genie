import json
from pathlib import Path

import numpy as np
import pandas as pd


def load_metadata(path: Path) -> dict:
    with path.open("r") as f:
        return json.load(f)


def load_data(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def compute_majority_reliance_scalar(df: pd.DataFrame) -> int:
    """
    Map evidence about children's reliance on social information / majority cues
    to a single Likert-style scalar in [-100, 100].

    We operationalize "reliance on social information and preference for majority cues"
    as the proportion of trials where children chose the majority option (code 2)
    among all demonstrated options (codes 2=majority, 3=minority).
    """
    # Only trials where the child chose either the majority or minority demonstrated option
    demo_mask = df["feature1"].isin([2, 3])
    demo_df = df.loc[demo_mask].copy()

    if demo_df.empty:
        # No informative data – return neutral
        return 0

    majority_prop = (demo_df["feature1"] == 2).mean()

    # Center at 0.5 (no preference), scale to [-100, 100]
    scalar = (majority_prop - 0.5) * 200.0

    # Clip to bounds and round to nearest integer
    scalar_int = int(np.round(np.clip(scalar, -100, 100)))
    return scalar_int


def main() -> None:
    cwd = Path(".")
    info_path = cwd / "info.json"
    boxes_path = cwd / "boxes.csv"

    info = load_metadata(info_path)
    df = load_data(boxes_path)

    # Compute overall scalar capturing degree of majority preference
    scalar = compute_majority_reliance_scalar(df)

    # Write scalar conclusion as required
    conclusion_path = cwd / "conclusion.txt"
    with conclusion_path.open("w") as f:
        f.write(str(scalar))


if __name__ == "__main__":
    main()

