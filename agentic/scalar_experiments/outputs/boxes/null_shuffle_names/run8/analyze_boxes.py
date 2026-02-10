import json
from pathlib import Path

import pandas as pd


def load_metadata() -> dict:
    info_path = Path("info.json")
    with info_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_data() -> pd.DataFrame:
    data_path = Path("boxes.csv")
    return pd.read_csv(data_path)


def summarize_choices(df: pd.DataFrame) -> dict:
    """
    Compute key summary statistics on majority-following behavior.
    """
    # Map majority_first codes: 1=unchosen, 2=majority, 3=minority
    choice_counts = df["majority_first"].value_counts().to_dict()

    majority_share = (df["majority_first"] == 2).mean()
    minority_share = (df["majority_first"] == 3).mean()
    undemonstrated_share = (df["majority_first"] == 1).mean()

    # Age-wise majority following
    age_majority = (
        df.assign(majority_choice=lambda d: d["majority_first"] == 2)
        .groupby("age")["majority_choice"]
        .mean()
        .to_dict()
    )

    # Site-wise majority following (y is site id)
    site_majority = (
        df.assign(majority_choice=lambda d: d["majority_first"] == 2)
        .groupby("y")["majority_choice"]
        .mean()
        .to_dict()
    )

    # Simple proxy for cultural context: culture column (0/1)
    culture_majority = (
        df.assign(majority_choice=lambda d: d["majority_first"] == 2)
        .groupby("culture")["majority_choice"]
        .mean()
        .to_dict()
    )

    return {
        "choice_counts": choice_counts,
        "majority_share": majority_share,
        "minority_share": minority_share,
        "undemonstrated_share": undemonstrated_share,
        "age_majority": age_majority,
        "site_majority": site_majority,
        "culture_majority": culture_majority,
    }


def main() -> None:
    info = load_metadata()
    df = load_data()
    summary = summarize_choices(df)

    print("Research question:")
    for q in info.get("research_questions", []):
        print("-", q)
    print()

    print("Overall choice counts (1=undemonstrated, 2=majority, 3=minority):")
    print(summary["choice_counts"])
    print()

    print("Overall shares:")
    print("majority_share", summary["majority_share"])
    print("minority_share", summary["minority_share"])
    print("undemonstrated_share", summary["undemonstrated_share"])
    print()

    print("Age-wise majority-following (proportion choosing majority):")
    for age, prop in sorted(summary["age_majority"].items()):
        print(f"age {age}: {prop:.3f}")
    print()

    print("Site-wise majority-following (proportion choosing majority):")
    for site, prop in sorted(summary["site_majority"].items()):
        print(f"site {site}: {prop:.3f}")
    print()

    print("Culture-wise majority-following (culture variable):")
    for culture, prop in sorted(summary["culture_majority"].items()):
        print(f"culture {culture}: {prop:.3f}")


if __name__ == "__main__":
    main()

