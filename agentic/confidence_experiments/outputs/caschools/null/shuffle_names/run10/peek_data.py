import pandas as pd
from pathlib import Path


def main() -> None:
    df = pd.read_csv(Path("caschools.csv"))
    print(df.dtypes)
    print("\nFirst few rows:")
    print(df.head())


if __name__ == "__main__":
    main()

