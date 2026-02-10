import pandas as pd


def main() -> None:
    df = pd.read_csv("boxes.csv")

    # Binary indicator of following the majority demonstration.
    df["majority_choice"] = (df["y"] == 2).astype(int)

    # Culture-level variation in majority-following.
    culture_rates = df.groupby("culture")["majority_choice"].mean()
    culture_sd = culture_rates.std()

    # Age-stage (developmental stage) variation in majority-following.
    age_bins = [0, 25, 35, 45, 100]
    age_labels = ["<=25", "26-35", "36-45", "46+"]
    df["age_stage"] = pd.cut(
        df["age"],
        bins=age_bins,
        labels=age_labels,
        right=True,
        include_lowest=True,
    )
    age_rates = df.groupby("age_stage", observed=False)["majority_choice"].mean()
    age_sd = age_rates.std()

    # Simple rule-based mapping from observed variation to Likert-style scalar:
    # - Larger standard deviations across cultures/ages indicate stronger evidence
    #   that reliance on majority cues varies with culture/developmental stage.
    if culture_sd > 0.07 or age_sd > 0.05:
        scalar = 60
    elif culture_sd > 0.05 or age_sd >= 0.035:
        scalar = 40
    elif culture_sd > 0.03 or age_sd > 0.02:
        scalar = 20
    else:
        scalar = 0

    # Positive values correspond to evidence that reliance on social/majority
    # information does vary across cultures and developmental stages.
    with open("conclusion.txt", "w", encoding="utf-8") as f:
        f.write(str(int(scalar)))


if __name__ == "__main__":
    main()

