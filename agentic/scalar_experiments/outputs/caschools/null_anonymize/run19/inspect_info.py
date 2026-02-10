import json


def main() -> None:
    with open("info.json", "r") as f:
        info = json.load(f)

    fields = info.get("data_desc", {}).get("fields", [])
    for field in fields:
        col = field.get("column")
        desc = field.get("properties", {}).get("description", "")
        print(f"{col}: {desc}")


if __name__ == "__main__":
    main()

