import importlib


def check(name: str) -> None:
    try:
        importlib.import_module(name)
        print(f"{name}: OK")
    except ImportError as exc:
        print(f"{name}: FAIL ({exc})")


def main() -> None:
    for pkg in ["numpy", "pandas", "statsmodels"]:
        check(pkg)


if __name__ == "__main__":
    main()

