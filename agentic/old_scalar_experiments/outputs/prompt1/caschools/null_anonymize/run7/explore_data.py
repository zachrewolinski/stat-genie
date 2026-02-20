import pandas as pd


def main() -> None:
    df = pd.read_csv("caschools.csv")
    # Basic info about key variables
    df["student_teacher_ratio"] = df["feature6"] / df["feature7"]
    df["testscr"] = (df["feature14"] + df["feature15"]) / 2

    summary = {
        "n_rows": len(df),
        "student_teacher_ratio": df["student_teacher_ratio"].describe().to_dict(),
        "testscr": df["testscr"].describe().to_dict(),
        "corr_str_testscr": df[["student_teacher_ratio", "testscr"]].corr().iloc[0, 1],
    }

    # Print in a simple, parseable way
    import json

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

