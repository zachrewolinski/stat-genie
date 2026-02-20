import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


DATA_PATH = Path("affairs.csv")


def main() -> None:
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=["affairs", "children"])
    df["any_affair"] = df["affairs"] > 0

    grouped = (
        df.groupby("children")
        .agg(
            n=("any_affair", "size"),
            any_affair_rate=("any_affair", "mean"),
            mean_affairs=("affairs", "mean"),
        )
        .reset_index()
    )

    affairs_with_children = df.loc[df["children"] == "yes", "affairs"]
    affairs_without_children = df.loc[df["children"] == "no", "affairs"]
    t_res = stats.ttest_ind(
        affairs_with_children,
        affairs_without_children,
        equal_var=False,
    )

    contingency = pd.crosstab(df["children"], df["any_affair"])
    chi2, chi2_p, dof, expected = stats.chi2_contingency(contingency)

    results = {
        "group_stats": grouped.to_dict(orient="records"),
        "t_test": {
            "statistic": float(t_res.statistic),
            "p_value": float(t_res.pvalue),
        },
        "chi2_test": {
            "chi2": float(chi2),
            "p_value": float(chi2_p),
            "dof": int(dof),
            "contingency": contingency.to_dict(),
        },
    }

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

