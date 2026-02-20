import json


def main() -> None:
    with open("info.json", "r", encoding="utf-8") as f:
        info = json.load(f)

    fields = info.get("data_desc", {}).get("fields", [])
    print("Column mapping from info.json:\n")
    for field in fields:
        col = field.get("column")
        props = field.get("properties", {})
        dtype = props.get("dtype")
        desc = props.get("description", "").strip()
        print(f"{col:9s} | {dtype:7s} | {desc}")


if __name__ == "__main__":
    main()

