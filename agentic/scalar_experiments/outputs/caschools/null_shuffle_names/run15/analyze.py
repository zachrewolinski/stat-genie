import pandas as pd
import numpy as np


def main() -> None:
    # Load data
    df = pd.read_csv("caschools.csv")

    # According to info.json descriptions (names are shuffled):
    # - "english" column stores total enrollment.
    # - "students" column stores number of teachers.
    # - "district" column stores average reading score.
    # - "expenditure" column stores average math score.
    enrollment = df["english"]
    teachers = df["students"]

    # Student–teacher ratio
    ratio = enrollment / teachers

    # Academic performance: mean of reading and math scores
    reading = df["district"]
    math = df["expenditure"]
    performance = (reading + math) / 2.0

    # Measure association via Pearson correlation
    corr = ratio.corr(performance)

    # Research question: "Is a lower student-teacher ratio associated with
    # higher academic performance?"
    # A negative correlation supports a "Yes" answer. Map to Likert -100..100
    # where 100 = very strong "Yes".
    scalar = int(np.rint(-corr * 100))
    scalar = int(np.clip(scalar, -100, 100))

    # Write the scalar conclusion to the required file, as the only content.
    with open("conclusion.txt", "w", encoding="utf-8") as f:
        f.write(f"{scalar}")


if __name__ == "__main__":
    main()

