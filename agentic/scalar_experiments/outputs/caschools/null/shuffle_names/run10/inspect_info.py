import json
from pathlib import Path


def main() -> None:
    info_path = Path("info.json")
    with info_path.open() as f:
        info = json.load(f)

    print("Research questions:")
    for q in info.get("research_questions", []):
        print("-", q)

    print("\nFields and descriptions:")
    for field in info.get("data_desc", {}).get("fields", []):
        col = field.get("column")
        props = field.get("properties", {})
        desc = props.get("description", "")
        dtype = props.get("dtype")
        print(f"{col:12s} | {dtype:8s} | {desc}")


if __name__ == "__main__":
    main()

