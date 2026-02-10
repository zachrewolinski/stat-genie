import pandas as pd
import numpy as np


def main() -> None:
    df = pd.read_csv("boxes.csv")

    # Map outcome: 2 = majority (positive), 1 = undemonstrated, 3 = minority
    majority_share = (df["y"] == 2).mean()

    # Basic cross-cultural / age sanity checks
    age_majority = df.groupby("age")["y"].apply(lambda s: (s == 2).mean())
    culture_majority = df.groupby("culture")["y"].apply(lambda s: (s == 2).mean())

    # Combine evidence: overall majority use, variability across age/culture.
    # Center majority_share around 0.5, scale to [-100, 100].
    evidence = (majority_share - 0.5) / 0.5
    scalar = int(np.clip(round(evidence * 80), -100, 100))

    # Bias slightly toward "yes" if most ages and cultures show majority > 0.5
    if (age_majority > 0.5).mean() > 0.6 and (culture_majority > 0.5).mean() > 0.6:
        scalar = min(100, scalar + 10)

    with open("conclusion.txt", "w", encoding="utf-8") as f:
        f.write(str(int(scalar)))


if __name__ == "__main__":
    main()

